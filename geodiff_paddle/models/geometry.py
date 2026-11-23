import paddle
from graph_ops import scatter_add


def get_distance(pos, edge_index):
    return paddle.linalg.norm(pos[edge_index[0]] - pos[edge_index[1]], axis=-1)


def eq_transform(score_d, pos, edge_index, edge_length):
    N = pos.shape[0]
    dd_dr = 1.0 / edge_length * (pos[edge_index[0]] - pos[edge_index[1]])
    score_pos = scatter_add(dd_dr * score_d, edge_index[0], dim=0, dim_size=N
        ) + scatter_add(-dd_dr * score_d, edge_index[1], dim=0, dim_size=N)
    return score_pos


def convert_cluster_score_d(cluster_score_d, cluster_pos,
    cluster_edge_index, cluster_edge_length, subgraph_index):
    """
    Args:
        cluster_score_d:    (E_c, 1)
        subgraph_index:     (N, )
    """
    cluster_score_pos = eq_transform(cluster_score_d, cluster_pos,
        cluster_edge_index, cluster_edge_length)
    score_pos = cluster_score_pos[subgraph_index]
    return score_pos


def get_angle(pos, angle_index):
    """
    Args:
        pos:  (N, 3)
        angle_index:  (3, A), left-center-right.
    """
    n1, ctr, n2 = angle_index
    v1 = pos[n1] - pos[ctr]
    v2 = pos[n2] - pos[ctr]
    inner_prod = paddle.sum(v1 * v2, axis=-1, keepdim=True)
    length_prod = (paddle.linalg.norm(v1, axis=-1, keepdim=True)
                   * paddle.linalg.norm(v2, axis=-1, keepdim=True))
    angle = paddle.acos(inner_prod / length_prod)
    return angle


def get_dihedral(pos, dihedral_index):
    """
    Args:
        pos:  (N, 3)
        dihedral:  (4, A)
    """
    n1, ctr1, ctr2, n2 = dihedral_index
    v_ctr = pos[ctr2] - pos[ctr1]
    v1 = pos[n1] - pos[ctr1]
    v2 = pos[n2] - pos[ctr2]
    n1 = paddle.cross(v_ctr, v1, axis=-1)
    n2 = paddle.cross(v_ctr, v2, axis=-1)
    inner_prod = paddle.sum(n1 * n2, axis=1, keepdim=True)
    length_prod = (paddle.linalg.norm(n1, axis=-1, keepdim=True)
                   * paddle.linalg.norm(n2, axis=-1, keepdim=True))
    dihedral = paddle.acos(inner_prod / length_prod)
    return dihedral
