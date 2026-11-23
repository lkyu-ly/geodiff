import paddle
from torch_scatter import scatter_add, scatter_mean
from torch_scatter import scatter
from torch_geometric.data import Data, Batch
import numpy as np
from numpy import pi as PI
from tqdm.auto import tqdm
from utils.chem import BOND_TYPES
from ..common import MultiLayerPerceptron, assemble_atom_pair_feature, generate_symmetric_edge_noise, extend_graph_order_radius
from ..encoder import SchNetEncoder, GINEncoder, get_edge_encoder
from ..geometry import get_distance, get_angle, get_dihedral, eq_transform
import pdb


def get_beta_schedule(beta_schedule, *, beta_start, beta_end,
    num_diffusion_timesteps):

    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)
    if beta_schedule == 'quad':
        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5,
            num_diffusion_timesteps, dtype=np.float64) ** 2
    elif beta_schedule == 'linear':
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps,
            dtype=np.float64)
    elif beta_schedule == 'const':
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'jsd':
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1,
            num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'sigmoid':
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class DualEncoderEpsNetwork(paddle.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        """
        edge_encoder:  Takes both edge type and edge length as input and outputs a vector
        [Note]: node embedding is done in SchNetEncoder
        """
        self.edge_encoder_global = get_edge_encoder(config)
        self.edge_encoder_local = get_edge_encoder(config)
        """
        timestep embedding
        """
        """
        The graph neural network that extracts node-wise features.
        """
        self.encoder_global = SchNetEncoder(hidden_channels=config.
            hidden_dim, num_filters=config.hidden_dim, num_interactions=
            config.num_convs, edge_channels=self.edge_encoder_global.
            out_channels, cutoff=config.cutoff, smooth=config.smooth_conv)
        self.encoder_local = GINEncoder(hidden_dim=config.hidden_dim,
            num_convs=config.num_convs_local)
        """
        `output_mlp` takes a mixture of two nodewise features and edge features as input and outputs 
            gradients w.r.t. edge_length (out_dim = 1).
        """
        self.grad_global_dist_mlp = MultiLayerPerceptron(2 * config.
            hidden_dim, [config.hidden_dim, config.hidden_dim // 2, 1],
            activation=config.mlp_act)
        self.grad_local_dist_mlp = MultiLayerPerceptron(2 * config.
            hidden_dim, [config.hidden_dim, config.hidden_dim // 2, 1],
            activation=config.mlp_act)
        """
        Incorporate parameters together
        """
        self.model_global = paddle.nn.ModuleList([self.edge_encoder_global,
            self.encoder_global, self.grad_global_dist_mlp])
        self.model_local = paddle.nn.ModuleList([self.edge_encoder_local,
            self.encoder_local, self.grad_local_dist_mlp])
        self.model_type = config.type
        if self.model_type == 'diffusion':
            betas = get_beta_schedule(beta_schedule=config.beta_schedule,
                beta_start=config.beta_start, beta_end=config.beta_end,
                num_diffusion_timesteps=config.num_diffusion_timesteps)
            betas = paddle.from_numpy(betas).float()
            self.betas = paddle.nn.Parameter(betas, requires_grad=False)
            alphas = (1.0 - betas).cumprod(dim=0)
            self.alphas = paddle.nn.Parameter(alphas, requires_grad=False)
            self.num_timesteps = self.betas.size(0)
        elif self.model_type == 'dsm':
            sigmas = paddle.tensor(np.exp(np.linspace(np.log(config.
                sigma_begin), np.log(config.sigma_end), config.
                num_noise_level)), dtype=paddle.float32)
            self.sigmas = paddle.nn.Parameter(sigmas, requires_grad=False)
            self.num_timesteps = self.sigmas.size(0)

    def forward(self, atom_type, pos, bond_index, bond_type, batch,
        time_step, edge_index=None, edge_type=None, edge_length=None,
        return_edges=False, extend_order=True, extend_radius=True,
        is_sidechain=None):
        """
        Args:
            atom_type:  Types of atoms, (N, ).
            bond_index: Indices of bonds (not extended, not radius-graph), (2, E).
            bond_type:  Bond types, (E, ).
            batch:      Node index to graph index, (N, ).
        """
        N = atom_type.size(0)
        if edge_index is None or edge_type is None or edge_length is None:
            edge_index, edge_type = extend_graph_order_radius(num_nodes=N,
                pos=pos, edge_index=bond_index, edge_type=bond_type, batch=
                batch, order=self.config.edge_order, cutoff=self.config.
                cutoff, extend_order=extend_order, extend_radius=
                extend_radius, is_sidechain=is_sidechain)
            edge_length = get_distance(pos, edge_index).unsqueeze(-1)
        local_edge_mask = is_local_edge(edge_type)
        if self.model_type == 'dsm':
            noise_levels = self.sigmas.index_select(0, time_step)
            node2graph = batch
            edge2graph = node2graph.index_select(0, edge_index[0])
            sigma_edge = noise_levels.index_select(0, edge2graph).unsqueeze(-1)
        elif self.model_type == 'diffusion':
            sigma_edge = paddle.ones(size=(edge_index.size(1), 1), device=
                pos.device)
        edge_attr_global = self.edge_encoder_global(edge_length=edge_length,
            edge_type=edge_type)
        node_attr_global = self.encoder_global(z=atom_type, edge_index=
            edge_index, edge_length=edge_length, edge_attr=edge_attr_global)
        h_pair_global = assemble_atom_pair_feature(node_attr=
            node_attr_global, edge_index=edge_index, edge_attr=edge_attr_global
            )
        edge_inv_global = self.grad_global_dist_mlp(h_pair_global) * (1.0 /
            sigma_edge)
        edge_attr_local = self.edge_encoder_global(edge_length=edge_length,
            edge_type=edge_type)
        node_attr_local = self.encoder_local(z=atom_type, edge_index=
            edge_index[:, local_edge_mask], edge_attr=edge_attr_local[
            local_edge_mask])
        h_pair_local = assemble_atom_pair_feature(node_attr=node_attr_local,
            edge_index=edge_index[:, local_edge_mask], edge_attr=
            edge_attr_local[local_edge_mask])
        if isinstance(sigma_edge, paddle.Tensor):
            edge_inv_local = self.grad_local_dist_mlp(h_pair_local) * (1.0 /
                sigma_edge[local_edge_mask])
        else:
            edge_inv_local = self.grad_local_dist_mlp(h_pair_local) * (1.0 /
                sigma_edge)
        if return_edges:
            return (edge_inv_global, edge_inv_local, edge_index, edge_type,
                edge_length, local_edge_mask)
        else:
            return edge_inv_global, edge_inv_local

    def get_loss(self, atom_type, pos, bond_index, bond_type, batch,
        num_nodes_per_graph, num_graphs, anneal_power=2.0,
        return_unreduced_loss=False, return_unreduced_edge_loss=False,
        extend_order=True, extend_radius=True, is_sidechain=None,
        record_inputs=None):
        if self.model_type == 'diffusion':
            return self.get_loss_diffusion(atom_type, pos, bond_index,
                bond_type, batch, num_nodes_per_graph, num_graphs,
                anneal_power, return_unreduced_loss,
                return_unreduced_edge_loss, extend_order, extend_radius,
                is_sidechain, record_inputs=record_inputs)
        elif self.model_type == 'dsm':
            return self.get_loss_dsm(atom_type, pos, bond_index, bond_type,
                batch, num_nodes_per_graph, num_graphs, anneal_power,
                return_unreduced_loss, return_unreduced_edge_loss,
                extend_order, extend_radius, is_sidechain)

    def get_loss_diffusion(self, atom_type, pos, bond_index, bond_type,
        batch, num_nodes_per_graph, num_graphs, anneal_power=2.0,
        return_unreduced_loss=False, return_unreduced_edge_loss=False,
        extend_order=True, extend_radius=True, is_sidechain=None,
        record_inputs=None):
        N = atom_type.size(0)
        node2graph = batch
        time_step = paddle.randint(low=0, high=self.num_timesteps, shape=(
            num_graphs // 2 + 1,))
        time_step = paddle.cat([time_step, self.num_timesteps - time_step -
            1], dim=0)[:num_graphs]
        a = self.alphas.index_select(0, time_step)
        a_pos = a.index_select(0, node2graph).unsqueeze(-1)
        pos_noise = paddle.zeros(size=pos.size(), device=pos.device)
        pos_noise.normal_()
        pos_perturbed = pos + pos_noise * (1.0 - a_pos).sqrt() / a_pos.sqrt()
        if record_inputs is not None:
            record_inputs(time_step, pos_noise)
        (edge_inv_global, edge_inv_local, edge_index, edge_type,
            edge_length, local_edge_mask) = (self(atom_type=atom_type, pos=
            pos_perturbed, bond_index=bond_index, bond_type=bond_type,
            batch=batch, time_step=time_step, return_edges=True,
            extend_order=extend_order, extend_radius=extend_radius,
            is_sidechain=is_sidechain))
        edge2graph = node2graph.index_select(0, edge_index[0])
        a_edge = a.index_select(0, edge2graph).unsqueeze(-1)
        d_gt = get_distance(pos, edge_index).unsqueeze(-1)
        d_perturbed = edge_length
        train_edge_mask = is_train_edge(edge_index, is_sidechain)
        d_perturbed = paddle.where(train_edge_mask.unsqueeze(-1),
            d_perturbed, d_gt)
        if self.config.edge_encoder == 'gaussian':
            d_sgn = paddle.sign(d_perturbed)
            d_perturbed = paddle.clamp(d_perturbed * d_sgn, min=0.01, max=
                float('inf'))
        d_target = (d_gt - d_perturbed) / (1.0 - a_edge).sqrt() * a_edge.sqrt()
        global_mask = paddle.logical_and(paddle.logical_or(d_perturbed <=
            self.config.cutoff, local_edge_mask.unsqueeze(-1)), ~
            local_edge_mask.unsqueeze(-1))
        target_d_global = paddle.where(global_mask, d_target, paddle.
            zeros_like(d_target))
        edge_inv_global = paddle.where(global_mask, edge_inv_global, paddle
            .zeros_like(edge_inv_global))
        target_pos_global = eq_transform(target_d_global, pos_perturbed,
            edge_index, edge_length)
        node_eq_global = eq_transform(edge_inv_global, pos_perturbed,
            edge_index, edge_length)
        loss_global = (node_eq_global - target_pos_global) ** 2
        loss_global = 2 * paddle.sum(loss_global, dim=-1, keepdim=True)
        target_pos_local = eq_transform(d_target[local_edge_mask],
            pos_perturbed, edge_index[:, local_edge_mask], edge_length[
            local_edge_mask])
        node_eq_local = eq_transform(edge_inv_local, pos_perturbed,
            edge_index[:, local_edge_mask], edge_length[local_edge_mask])
        loss_local = (node_eq_local - target_pos_local) ** 2
        loss_local = 5 * paddle.sum(loss_local, dim=-1, keepdim=True)
        loss = loss_global + loss_local
        if return_unreduced_edge_loss:
            pass
        elif return_unreduced_loss:
            return loss, loss_global, loss_local
        else:
            return loss

    def langevin_dynamics_sample(self, atom_type, pos_init, bond_index,
        bond_type, batch, num_graphs, extend_order, extend_radius=True,
        n_steps=100, step_lr=1e-06, clip=1000, clip_local=None, clip_pos=
        None, min_sigma=0, is_sidechain=None, global_start_sigma=float(
        'inf'), w_global=0.2, w_reg=1.0, **kwargs):
        if self.model_type == 'diffusion':
            return self.langevin_dynamics_sample_diffusion(atom_type,
                pos_init, bond_index, bond_type, batch, num_graphs,
                extend_order, extend_radius, n_steps, step_lr, clip,
                clip_local, clip_pos, min_sigma, is_sidechain,
                global_start_sigma, w_global, w_reg, sampling_type=kwargs.
                get('sampling_type', 'ddpm_noisy'), eta=kwargs.get('eta', 1.0))
        elif self.model_type == 'dsm':
            return self.langevin_dynamics_sample_dsm(atom_type, pos_init,
                bond_index, bond_type, batch, num_graphs, extend_order,
                extend_radius, n_steps, step_lr, clip, clip_local, clip_pos,
                min_sigma, is_sidechain, global_start_sigma, w_global, w_reg)

    def langevin_dynamics_sample_diffusion(self, atom_type, pos_init,
        bond_index, bond_type, batch, num_graphs, extend_order,
        extend_radius=True, n_steps=100, step_lr=1e-06, clip=1000,
        clip_local=None, clip_pos=None, min_sigma=0, is_sidechain=None,
        global_start_sigma=float('inf'), w_global=0.2, w_reg=1.0, **kwargs):

        def compute_alpha(beta, t):
            beta = paddle.cat([paddle.zeros(1).to(beta.device), beta], dim=0)
            a = (1 - beta).cumprod(dim=0).index_select(0, t + 1)
            return a
        sigmas = (1.0 - self.alphas).sqrt() / self.alphas.sqrt()
        pos_traj = []
        if is_sidechain is not None:
            assert pos_gt is not None, 'need crd of backbone for sidechain prediction'
        with paddle.no_grad():
            seq = range(self.num_timesteps - n_steps, self.num_timesteps)
            seq_next = [-1] + list(seq[:-1])
            pos = pos_init * sigmas[-1]
            if is_sidechain is not None:
                pos[~is_sidechain] = pos_gt[~is_sidechain]
            for i, j in tqdm(zip(reversed(seq), reversed(seq_next)), desc=
                'sample'):
                t = paddle.full(size=(num_graphs,), fill_value=i, dtype=
                    paddle.long, device=pos.device)
                (edge_inv_global, edge_inv_local, edge_index, edge_type,
                    edge_length, local_edge_mask) = (self(atom_type=
                    atom_type, pos=pos, bond_index=bond_index, bond_type=
                    bond_type, batch=batch, time_step=t, return_edges=True,
                    extend_order=extend_order, extend_radius=extend_radius,
                    is_sidechain=is_sidechain))
                node_eq_local = eq_transform(edge_inv_local, pos,
                    edge_index[:, local_edge_mask], edge_length[
                    local_edge_mask])
                if clip_local is not None:
                    node_eq_local = clip_norm(node_eq_local, limit=clip_local)
                if sigmas[i] < global_start_sigma:
                    edge_inv_global = edge_inv_global * (1 -
                        local_edge_mask.view(-1, 1).float())
                    node_eq_global = eq_transform(edge_inv_global, pos,
                        edge_index, edge_length)
                    node_eq_global = clip_norm(node_eq_global, limit=clip)
                else:
                    node_eq_global = 0
                eps_pos = node_eq_local + node_eq_global * w_global
                sampling_type = kwargs.get('sampling_type', 'ddpm_noisy')
                noise = paddle.randn_like(pos)
                if (sampling_type == 'generalized' or sampling_type ==
                    'ddpm_noisy'):
                    b = self.betas
                    t = t[0]
                    next_t = (paddle.ones(1) * j).to(pos.device)
                    at = compute_alpha(b, t.long())
                    at_next = compute_alpha(b, next_t.long())
                    if sampling_type == 'generalized':
                        eta = kwargs.get('eta', 1.0)
                        et = -eps_pos
                        c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 -
                            at)).sqrt()
                        c2 = (1 - at_next - c1 ** 2).sqrt()
                        step_size_pos_ld = step_lr * (sigmas[i] / 0.01
                            ) ** 2 / sigmas[i]
                        step_size_pos_generalized = 5 * ((1 - at).sqrt() /
                            at.sqrt() - c2 / at_next.sqrt())
                        step_size_pos = (step_size_pos_ld if 
                            step_size_pos_ld < step_size_pos_generalized else
                            step_size_pos_generalized)
                        step_size_noise_ld = paddle.sqrt(step_lr * (sigmas[
                            i] / 0.01) ** 2 * 2)
                        step_size_noise_generalized = 3 * (c1 / at_next.sqrt())
                        step_size_noise = (step_size_noise_ld if 
                            step_size_noise_ld <
                            step_size_noise_generalized else
                            step_size_noise_generalized)
                        pos_next = (pos - et * step_size_pos + noise *
                            step_size_noise)
                    elif sampling_type == 'ddpm_noisy':
                        atm1 = at_next
                        beta_t = 1 - at / atm1
                        e = -eps_pos
                        pos0_from_e = (1.0 / at).sqrt() * pos - (1.0 / at - 1
                            ).sqrt() * e
                        mean_eps = (atm1.sqrt() * beta_t * pos0_from_e + (1 -
                            beta_t).sqrt() * (1 - atm1) * pos) / (1.0 - at)
                        mean = mean_eps
                        mask = 1 - (t == 0).float()
                        logvar = beta_t.log()
                        pos_next = mean + mask * paddle.exp(0.5 * logvar
                            ) * noise
                elif sampling_type == 'ld':
                    step_size = step_lr * (sigmas[i] / 0.01) ** 2
                    pos_next = pos + step_size * eps_pos / sigmas[i
                        ] + noise * paddle.sqrt(step_size * 2)
                pos = pos_next
                if is_sidechain is not None:
                    pos[~is_sidechain] = pos_gt[~is_sidechain]
                if paddle.isnan(pos).any():
                    print('NaN detected. Please restart.')
                    raise FloatingPointError()
                pos = center_pos(pos, batch)
                if clip_pos is not None:
                    pos = paddle.clamp(pos, min=-clip_pos, max=clip_pos)
                pos_traj.append(pos.clone().cpu())
        return pos, pos_traj

    def get_loss_dsm(self, atom_type, pos, bond_index, bond_type, batch,
        num_nodes_per_graph, num_graphs, anneal_power=2.0,
        return_unreduced_loss=False, return_unreduced_edge_loss=False,
        extend_order=True, extend_radius=True, is_sidechain=None):
        N = atom_type.size(0)
        node2graph = batch
        time_step = paddle.randint(low=0, high=self.num_timesteps, shape=(
            num_graphs // 2 + 1,))
        time_step = paddle.cat([time_step, self.num_timesteps - time_step -
            1], dim=0)[:num_graphs]
        noise_levels = self.sigmas.index_select(0, time_step)
        sigmas_pos = noise_levels.index_select(0, node2graph).unsqueeze(-1)
        pos_noise = paddle.zeros(size=pos.size(), device=pos.device)
        pos_noise.normal_()
        pos_perturbed = pos + pos_noise * sigmas_pos
        (edge_inv_global, edge_inv_local, edge_index, edge_type,
            edge_length, local_edge_mask) = (self(atom_type=atom_type, pos=
            pos_perturbed, bond_index=bond_index, bond_type=bond_type,
            batch=batch, time_step=time_step, return_edges=True,
            extend_order=extend_order, extend_radius=extend_radius,
            is_sidechain=is_sidechain))
        edge2graph = node2graph.index_select(0, edge_index[0])
        sigmas_edge = noise_levels.index_select(0, edge2graph).unsqueeze(-1)
        d_gt = get_distance(pos, edge_index).unsqueeze(-1)
        d_perturbed = edge_length
        train_edge_mask = is_train_edge(edge_index, is_sidechain)
        d_perturbed = paddle.where(train_edge_mask.unsqueeze(-1),
            d_perturbed, d_gt)
        if self.config.edge_encoder == 'gaussian':
            d_sgn = paddle.sign(d_perturbed)
            d_perturbed = paddle.clamp(d_perturbed * d_sgn, min=0.01, max=
                float('inf'))
        d_target = 1.0 / sigmas_edge ** 2 * (d_gt - d_perturbed)
        global_mask = paddle.logical_and(paddle.logical_or(d_perturbed <=
            self.config.cutoff, local_edge_mask.unsqueeze(-1)), ~
            local_edge_mask.unsqueeze(-1))
        target_d_global = paddle.where(global_mask, d_target, paddle.
            zeros_like(d_target))
        edge_inv_global = paddle.where(global_mask, edge_inv_global, paddle
            .zeros_like(edge_inv_global))
        target_pos_global = eq_transform(target_d_global, pos_perturbed,
            edge_index, edge_length)
        node_eq_global = eq_transform(edge_inv_global, pos_perturbed,
            edge_index, edge_length)
        loss_global = 0.5 * (node_eq_global - target_pos_global
            ) ** 2 * sigmas_pos ** anneal_power
        loss_global = 2 * paddle.sum(loss_global, dim=-1, keepdim=True)
        target_pos_local = eq_transform(d_target[local_edge_mask],
            pos_perturbed, edge_index[:, local_edge_mask], edge_length[
            local_edge_mask])
        node_eq_local = eq_transform(edge_inv_local, pos_perturbed,
            edge_index[:, local_edge_mask], edge_length[local_edge_mask])
        loss_local = 0.5 * (node_eq_local - target_pos_local
            ) ** 2 * sigmas_pos ** anneal_power
        loss_local = 5 * paddle.sum(loss_local, dim=-1, keepdim=True)
        loss = loss_global + loss_local
        if return_unreduced_edge_loss:
            pass
        elif return_unreduced_loss:
            return loss, loss_global, loss_local
        else:
            return loss

    def langevin_dynamics_sample_dsm(self, atom_type, pos_init, bond_index,
        bond_type, batch, num_graphs, extend_order, extend_radius=True,
        n_steps=100, step_lr=1e-06, clip=1000, clip_local=None, clip_pos=
        None, min_sigma=0, is_sidechain=None, global_start_sigma=float(
        'inf'), w_global=0.2, w_reg=1.0):
        sigmas = self.sigmas
        pos_traj = []
        if is_sidechain is not None:
            assert pos_gt is not None, 'need crd of backbone for sidechain prediction'
        with paddle.no_grad():
            pos = pos_init
            if is_sidechain is not None:
                pos[~is_sidechain] = pos_gt[~is_sidechain]
            for i, sigma in enumerate(tqdm(sigmas, desc='sample')):
                if sigma < min_sigma:
                    break
                time_step = paddle.full(size=(num_graphs,), fill_value=i,
                    dtype=paddle.long, device=pos.device)
                step_size = step_lr * (sigma / sigmas[-1]) ** 2
                for step in range(n_steps):
                    (edge_inv_global, edge_inv_local, edge_index, edge_type,
                        edge_length, local_edge_mask) = (self(atom_type=
                        atom_type, pos=pos, bond_index=bond_index,
                        bond_type=bond_type, batch=batch, time_step=
                        time_step, return_edges=True, extend_order=
                        extend_order, extend_radius=extend_radius,
                        is_sidechain=is_sidechain))
                    node_eq_local = eq_transform(edge_inv_local, pos,
                        edge_index[:, local_edge_mask], edge_length[
                        local_edge_mask])
                    if clip_local is not None:
                        node_eq_local = clip_norm(node_eq_local, limit=
                            clip_local)
                    if sigma < global_start_sigma:
                        edge_inv_global = edge_inv_global * (1 -
                            local_edge_mask.view(-1, 1).float())
                        node_eq_global = eq_transform(edge_inv_global, pos,
                            edge_index, edge_length)
                        node_eq_global = clip_norm(node_eq_global, limit=clip)
                    else:
                        node_eq_global = 0
                    eps_pos = node_eq_local + node_eq_global * w_global
                    noise = paddle.randn_like(pos) * paddle.sqrt(step_size * 2)
                    pos = pos + step_size * eps_pos + noise
                    if is_sidechain is not None:
                        pos[~is_sidechain] = pos_gt[~is_sidechain]
                    if paddle.isnan(pos).any():
                        print('NaN detected. Please restart.')
                        raise FloatingPointError()
                    pos = center_pos(pos, batch)
                    if clip_pos is not None:
                        pos = paddle.clamp(pos, min=-clip_pos, max=clip_pos)
                    pos_traj.append(pos.clone().cpu())
        return pos, pos_traj


def is_bond(edge_type):
    return paddle.logical_and(edge_type < len(BOND_TYPES), edge_type > 0)


def is_angle_edge(edge_type):
    return edge_type == len(BOND_TYPES) + 1 - 1


def is_dihedral_edge(edge_type):
    return edge_type == len(BOND_TYPES) + 2 - 1


def is_radius_edge(edge_type):
    return edge_type == 0


def is_local_edge(edge_type):
    return edge_type > 0


def is_train_edge(edge_index, is_sidechain):
    if is_sidechain is None:
        return paddle.ones(edge_index.size(1), device=edge_index.device).bool()
    else:
        is_sidechain = is_sidechain.bool()
        return paddle.logical_or(is_sidechain[edge_index[0]], is_sidechain[
            edge_index[1]])


def regularize_bond_length(edge_type, edge_length, rng=5.0):
    mask = is_bond(edge_type).float().reshape(-1, 1)
    d = -paddle.clamp(edge_length - rng, min=0.0, max=float('inf')) * mask
    return d


def center_pos(pos, batch):
    pos_center = pos - scatter_mean(pos, batch, dim=0)[batch]
    return pos_center


def clip_norm(vec, limit, p=2):
    norm = paddle.norm(vec, dim=-1, p=2, keepdim=True)
    denom = paddle.where(norm > limit, limit / norm, paddle.ones_like(norm))
    return vec * denom
