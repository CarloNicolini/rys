"""Lightning modules for the RYS training families."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rys.coloring_data import soft_coloring_loss, verify_coloring_tensor
from rys.nqueens_data import soft_nqueens_loss, verify_boards_tensor
from rys.sat_data import soft_sat_loss, verify_assignment_tensor
from rys.sorted_translation_data import sortedness, verify_sorted_tensor
from rys.tiny_transformer import FactorizedCNFSatTransformer
from rys.training.base import RysLitModule


class ClassifierLitModule(RysLitModule):
    primary_metric = "accuracy"
    primary_mode = "max"

    def _shared_step(self, batch, stage):
        # ``SatDataset`` carries BOTH the flat ``input_ids`` and the factorized
        # fields, so the batch alone cannot disambiguate the architecture; the
        # model type does. ``FactorizedCNFSatTransformer`` is keyword-only and
        # would reject a positional ``input_ids``.
        if isinstance(self.net, FactorizedCNFSatTransformer):
            out = self.net(
                variable_ids=batch["variable_ids"],
                sign_ids=batch["sign_ids"],
                clause_ids=batch["clause_ids"],
                slot_ids=batch["slot_ids"],
                token_type_ids=batch["token_type_ids"],
                attention_mask=batch["factor_attention_mask"],
                labels=batch["labels"],
            )
        else:
            out = self.net(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
        logits = out["logits"]
        loss = out["loss"]
        accuracy = (logits.argmax(dim=-1) == batch["labels"]).float().mean()
        return loss, {"accuracy": accuracy}


class AssignmentLitModule(RysLitModule):
    primary_metric = "valid_assignment_rate"
    primary_mode = "max"

    def __init__(
        self,
        net,
        *,
        lr: float,
        weight_decay: float,
        sat_loss_weight: float = 1.0,
        ce_loss_weight: float = 0.0,
        model_config=None,
    ) -> None:
        super().__init__(
            net,
            lr=lr,
            weight_decay=weight_decay,
            model_config=model_config,
            extra_hparams={
                "sat_loss_weight": sat_loss_weight,
                "ce_loss_weight": ce_loss_weight,
            },
        )
        self.sat_loss_weight = sat_loss_weight
        self.ce_loss_weight = ce_loss_weight

    def _shared_step(self, batch, stage):
        out = self.net(
            variable_ids=batch["variable_ids"],
            sign_ids=batch["sign_ids"],
            clause_ids=batch["clause_ids"],
            slot_ids=batch["slot_ids"],
            token_type_ids=batch["token_type_ids"],
            attention_mask=batch["factor_attention_mask"],
            assignment_labels=batch["assignment_labels"],
        )
        logits = out["logits"]
        cvi, csi, cm = _sat_clause_tensors(batch)
        loss = self.sat_loss_weight * soft_sat_loss(logits, cvi, csi, cm)
        if self.ce_loss_weight > 0:
            ce = F.cross_entropy(
                logits.reshape(-1, 2),
                batch["assignment_labels"].reshape(-1),
                ignore_index=-100,
            )
            loss = loss + self.ce_loss_weight * ce
        return loss, _assignment_metrics(logits, batch)


class SatMPLitModule(RysLitModule):
    primary_metric = "valid_assignment_rate"
    primary_mode = "max"

    def __init__(
        self,
        net,
        *,
        lr: float,
        weight_decay: float,
        deep_supervision: bool = True,
        ce_weight: float = 0.0,
        model_config=None,
    ) -> None:
        super().__init__(
            net,
            lr=lr,
            weight_decay=weight_decay,
            model_config=model_config,
            extra_hparams={"deep_supervision": deep_supervision, "ce_weight": ce_weight},
        )
        self.deep_supervision = deep_supervision
        self.ce_weight = ce_weight

    def _shared_step(self, batch, stage):
        cvi, csi, cm = _sat_clause_tensors(batch)
        out = self.net(
            clause_variable_ids=cvi,
            clause_sign_ids=csi,
            clause_mask=cm,
            return_round_logits=self.deep_supervision,
        )
        logits = out["logits"]
        round_logits = out["round_logits"] or []
        if self.deep_supervision and round_logits:
            loss = torch.stack([soft_sat_loss(logit, cvi, csi, cm) for logit in round_logits]).mean()
        else:
            loss = soft_sat_loss(logits, cvi, csi, cm)
        if self.ce_weight > 0:
            labels = batch["assignment_labels"]
            ce = F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1), ignore_index=-100)
            if torch.isfinite(ce):
                loss = loss + self.ce_weight * ce
        return loss, _assignment_metrics(logits, batch)


class SatStochasticLitModule(SatMPLitModule):
    def __init__(
        self,
        net,
        *,
        lr: float,
        weight_decay: float,
        sigma_floor: float,
        sigma_reg: float,
        stochastic_sample: bool,
        model_config=None,
    ) -> None:
        super().__init__(
            net,
            lr=lr,
            weight_decay=weight_decay,
            deep_supervision=True,
            model_config=model_config,
        )
        self.sigma_floor = sigma_floor
        self.sigma_reg = sigma_reg
        self.stochastic_sample = stochastic_sample

    def _shared_step(self, batch, stage):
        cvi, csi, cm = _sat_clause_tensors(batch)
        out = self.net(
            clause_variable_ids=cvi,
            clause_sign_ids=csi,
            clause_mask=cm,
            return_round_logits=True,
            sample=self.stochastic_sample,
        )
        logits = out["logits"]
        loss = torch.stack([soft_sat_loss(logit, cvi, csi, cm) for logit in out["round_logits"]]).mean()
        if out["mean_log_sigma"] is not None:
            sigma = out["mean_log_sigma"].exp()
            loss = loss + self.sigma_reg * torch.relu(torch.as_tensor(self.sigma_floor, device=sigma.device) - sigma)
        return loss, _assignment_metrics(logits, batch)


class SatCoverageLitModule(RysLitModule):
    primary_metric = "valid_assignment_rate"
    primary_mode = "max"

    def __init__(
        self,
        net,
        *,
        lr: float,
        weight_decay: float,
        regime: str,
        train_k: int,
        floor_sigma: float,
        floor_reg: float,
        diversity_weight: float,
        model_config=None,
    ) -> None:
        super().__init__(
            net,
            lr=lr,
            weight_decay=weight_decay,
            model_config=model_config,
            extra_hparams={
                "regime": regime,
                "train_k": train_k,
                "floor_sigma": floor_sigma,
                "floor_reg": floor_reg,
                "diversity_weight": diversity_weight,
            },
        )
        self.regime = regime
        self.train_k = train_k
        self.floor_sigma = floor_sigma
        self.floor_reg = floor_reg
        self.diversity_weight = diversity_weight

    def _shared_step(self, batch, stage):
        cvi, csi, cm = _sat_clause_tensors(batch)
        sample_losses = []
        sample_probs = []
        sigma_terms = []
        final_logits = None
        n_samples = self.train_k if stage == "train" else 1
        for _ in range(n_samples):
            out = self.net(clause_variable_ids=cvi, clause_sign_ids=csi, clause_mask=cm, sample=True)
            final_logits = out["logits"]
            sample_losses.append(soft_sat_loss(final_logits, cvi, csi, cm, reduction="none"))
            sample_probs.append(final_logits.softmax(dim=-1)[..., 1])
            if out["mean_log_sigma"] is not None:
                sigma_terms.append(out["mean_log_sigma"])
        stacked = torch.stack(sample_losses)
        loss = stacked.min(dim=0).values.mean() if self.regime == "best_of_k" else stacked.mean()
        if self.diversity_weight > 0 and len(sample_probs) > 1:
            loss = loss - self.diversity_weight * _diversity_reward(
                torch.stack(sample_probs),
                batch["assignment_mask"],
            )
        if self.regime == "floor" and sigma_terms:
            sigma = torch.stack(sigma_terms).mean().exp()
            loss = loss + self.floor_reg * torch.relu(torch.as_tensor(self.floor_sigma, device=sigma.device) - sigma)
        return loss, _assignment_metrics(final_logits, batch)


class ColoringMPLitModule(RysLitModule):
    primary_metric = "valid_coloring_rate"
    primary_mode = "max"

    def __init__(self, net, *, lr: float, weight_decay: float, given_ce_weight: float, model_config=None) -> None:
        super().__init__(
            net,
            lr=lr,
            weight_decay=weight_decay,
            model_config=model_config,
            extra_hparams={"given_ce_weight": given_ce_weight},
        )
        self.given_ce_weight = given_ce_weight

    def _shared_step(self, batch, stage):
        out = self.net(
            given_color=batch["given_color"],
            is_given=batch["is_given"],
            degree=batch["degree"],
            adjacency=batch["adjacency"],
            vertex_mask=batch["vertex_mask"],
            return_round_logits=True,
        )
        loss = torch.stack(
            [
                soft_coloring_loss(
                    logits,
                    batch["adjacency"],
                    batch["vertex_mask"],
                    given_color=batch["given_color"],
                    is_given=batch["is_given"],
                    given_ce_weight=self.given_ce_weight,
                )
                for logits in out["round_logits"]
            ]
        ).mean()
        logits = out["logits"]
        preds = logits.argmax(dim=-1)
        labels = batch["labels"]
        mask = batch["vertex_mask"]
        valid = verify_coloring_tensor(
            preds,
            batch["vertex_mask"],
            batch["adjacency"],
            batch["given_color"],
            batch["is_given"],
        )
        return loss, {
            "valid_coloring_rate": valid.float().mean(),
            "cell_accuracy": (((preds == labels) & mask).sum() / mask.sum().clamp_min(1)).float(),
        }


class NqueensMPLitModule(RysLitModule):
    primary_metric = "valid_board_rate"
    primary_mode = "max"

    def __init__(self, net, *, lr: float, weight_decay: float, given_ce_weight: float, model_config=None) -> None:
        super().__init__(
            net,
            lr=lr,
            weight_decay=weight_decay,
            model_config=model_config,
            extra_hparams={"given_ce_weight": given_ce_weight},
        )
        self.given_ce_weight = given_ce_weight

    def _shared_step(self, batch, stage):
        out = self.net(
            given_col=batch["given_col"],
            is_given=batch["is_given"],
            row_mask=batch["row_mask"],
            col_mask=batch["col_mask"],
            return_round_logits=True,
        )
        loss = torch.stack(
            [
                soft_nqueens_loss(
                    logits,
                    batch["row_mask"],
                    batch["col_mask"],
                    given_col=batch["given_col"],
                    is_given=batch["is_given"],
                    given_ce_weight=self.given_ce_weight,
                )
                for logits in out["round_logits"]
            ]
        ).mean()
        logits = out["logits"]
        preds = logits.argmax(dim=-1)
        labels = batch["labels"]
        mask = batch["row_mask"]
        valid = verify_boards_tensor(preds, batch["row_mask"], batch["given_col"], batch["is_given"])
        return loss, {
            "valid_board_rate": valid.float().mean(),
            "cell_accuracy": (((preds == labels) & mask).sum() / mask.sum().clamp_min(1)).float(),
        }


class SortedTranslationLitModule(RysLitModule):
    primary_metric = "sequence_accuracy"
    primary_mode = "max"

    def _shared_step(self, batch, stage):
        if isinstance(batch, list | tuple):
            outputs = [self._single_step(item) for item in batch]
            losses = torch.stack([loss for loss, _ in outputs])
            metrics = {
                key: torch.stack([item_metrics[key] for _, item_metrics in outputs]).mean()
                for key in outputs[0][1]
            }
            return losses.mean(), metrics
        if "input_ids" not in batch:
            outputs = [self._single_step(item) for item in batch.values()]
            losses = torch.stack([loss for loss, _ in outputs])
            metrics = {
                key: torch.stack([item_metrics[key] for _, item_metrics in outputs]).mean()
                for key in outputs[0][1]
            }
            return losses.mean(), metrics
        return self._single_step(batch)

    def _single_step(self, batch):
        out = self.net(batch["input_ids"], target_ids=batch["target_ids"])
        logits = out["logits"]
        loss = out["loss"]
        preds = logits.argmax(dim=-1)
        token_accuracy = (preds == batch["target_ids"]).float().mean()
        sequence_accuracy = verify_sorted_tensor(preds, batch["input_ids"]).float().mean()
        metrics = {
            "token_accuracy": token_accuracy,
            "sequence_accuracy": sequence_accuracy,
            "sortedness": sortedness(preds).mean(),
        }
        if hasattr(self.net, "model") and hasattr(self.net.model, "embed"):
            metrics["w_e_norm"] = self.net.model.embed.weight.norm().detach()
        if hasattr(self.net, "unembed"):
            metrics["w_u_norm"] = self.net.unembed.weight.norm().detach()
        return loss, metrics


def _sat_clause_tensors(batch):
    return batch["clause_variable_ids"], batch["clause_sign_ids"], batch["clause_mask"]


def _assignment_metrics(logits, batch):
    labels = batch["assignment_labels"]
    mask = batch["assignment_mask"]
    cvi, csi, cm = _sat_clause_tensors(batch)
    preds = logits.argmax(dim=-1)
    valid = verify_assignment_tensor(preds, cvi, csi, cm)
    return {
        "bit_accuracy": (((preds == labels) & mask).sum() / mask.sum().clamp_min(1)).float(),
        "exact_match": (((preds == labels) | ~mask).all(dim=1)).float().mean(),
        "valid_assignment_rate": valid.float().mean(),
    }


def _diversity_reward(prob_samples: torch.Tensor, assignment_mask: torch.Tensor) -> torch.Tensor:
    var = prob_samples.var(dim=0, unbiased=False)
    return (var * assignment_mask).sum() / assignment_mask.sum().clamp_min(1)
