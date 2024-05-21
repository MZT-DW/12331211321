import random
import time

import numpy as np
import math

import PyGP
from PyGP import Program, PopSemantic, TreeNode, TR


def r_snodes_select(smt_len, num):
    slts = rd.choice(range(smt_len), size=num, replace=False)
    return np.sort(slts)

def indivSelect_sem_4(tsematic_, candidates, tgdrvt_, tgdrvt_origin_, candidate_origin_,
                      depth_limit, mask, s3_size, tr_origin, org):  # 用于语义的个体选择
    candidate_ = [candidates[i].semantic for i in range(len(candidates))]
    tsematic = tsematic_ * mask
    candidate = candidate_ * mask
    tgdrvt = tgdrvt_ * mask
    # 选一个最近点
    idx_min = [-1, -1]
    
    cdd_mean_list = list(map(lambda x: np.mean(x), candidate))
    y_mean = np.mean(tsematic)
    b_list = list(map(lambda x: np.cov(candidate[x], tsematic)[0][1] / np.var(candidate[x]),#candidate[x] - cdd_mean_list[x]) * (tsematic - y_mean) / ((tsematic - y_mean) * (tsematic - y_mean)), 
                      range(len(candidate))))
    a_list = list(map(lambda x: y_mean - b_list[x] * cdd_mean_list[x], range(len(candidate))))


    rsdls_ = list(map(lambda x: np.subtract(tsematic, candidate[x] * b_list[x] + a_list[x]), range(len(candidate))))
    # dis_all_w = list(map(lambda x: np.sqrt(np.dot(x * tgdrvt_, x)), rsdls_))  # 加权距离
    dis_all_w = list(map(lambda x: np.sqrt(np.dot(x, x)), rsdls_))  # 加权距离
    dis_sorted = np.argsort(dis_all_w)
    succeed = False
    for i in range(len(candidate)):
        if candidates[dis_sorted[i]].tree.inner_size + 4 < PyGP.MAX_TREE_SIZE:
                candidate_min = [candidate[dis_sorted[i]], candidate[len(candidate) - 1]]
                idx_min = [int(dis_sorted[i]), int(len(candidate) - 1)]
                k = [b_list[dis_sorted[i]], a_list[dis_sorted[i]]]
                succeed = True
                break
    
    # 返回该两个点
    return (idx_min, candidate_min, None, k, succeed)


def indivSelect_sem_5(tsematic_, candidates, tgdrvt_, tgdrvt_origin_, candidate_origin_,
                      depth_limit, mask, s3_size, tr_origin, org):  # 用于语义的个体选择
    candidate_ = [candidates[i].semantic for i in range(len(candidates))]
    tsematic = tsematic_ * mask
    candidate = candidate_ * mask
    tgdrvt = tgdrvt_ * mask
    # 选一个最近点
    idx_min = [-1, -1]

    rsdls_ = list(map(lambda x: np.subtract(tsematic, candidate[x]), range(len(candidate))))
    # dis_all_w = list(map(lambda x: np.sqrt(np.dot(x * tgdrvt_, x)), rsdls_))  # 加权距离
    dis_all_w = list(map(lambda x: np.sqrt(np.dot(x, x)), rsdls_))  # 加权距离
    dis_sorted = np.argsort(dis_all_w)
    succeed = False
    for i in range(len(candidate)):
        if candidates[dis_sorted[i]].tree.inner_size < PyGP.MAX_TREE_SIZE:
                candidate_min = candidate[dis_sorted[i]]
                idx_min = int(dis_sorted[i])
                k = 1.
                succeed = True
                break
    print('here?????')
    # 返回该两个点
    return (idx_min, candidate_min, None, k, succeed)


def Levenberg_Marquarelt_2(tgdrvt, tgdrvt_origin, tsematic, candidate_1, candidate_2, time, init_k=None, r_=None):
    count = 0
    if not init_k:
        print("!!!!!!!!!!!!!!!!!!!")
        k = least_square_method(tsematic, candidate_1, candidate_2, tgdrvt)
        k0 = 1 - k
        k1 = k
    else:
        k0 = init_k[0]
        k1 = init_k[1]

    tsematic = np.array(tsematic)
    candidate_1 = np.array(candidate_1)
    candidate_2 = np.array(candidate_2)
    (tgdrvt, tgdrvt_origin) = (np.array(tgdrvt), np.array(tgdrvt_origin))
    # if (any(np.isnan(tgdrvt)) or any(np.isinf(tgdrvt))or any(np.isnan(candidate_1)) or any(np.isnan(candidate_2)) or any(np.isinf(candidate_1)) or any(np.isinf(candidate_2))):
    #     print((any(np.isnan(tgdrvt)), any(np.isinf(tgdrvt)), any(np.isnan(candidate_1)), any(np.isnan(candidate_2)), any(np.isinf(candidate_1)), any(np.isinf(candidate_2))))
    #     print(tgdrvt)
    #     assert (0 == 1)
    cdd = k0 * candidate_1 + k1 * candidate_2

    vec = np.subtract(cdd, tsematic)

    vec_last = np.dot(vec * tgdrvt, vec)
    vec_last_1 = np.dot(vec, vec)
    vec_best = vec_last
    vec_best_1 = vec_last_1

    k0_best = float(k0)
    k1_best = float(k1)

    # x_tmp = cdd - np.sqrt(np.dot(cdd, cdd))
    # if r_:
    #     x1 = np.sqrt(np.dot(x_tmp, x_tmp))
    #     vec_last = np.dot(x_tmp, r_[0]) / (x1 * r_[1]) if (x1 != 0 and r_[1] != 0) else (1 if x1 == r_[1] else -1)
    #     vec_best = vec_best_1 = vec_last
    # else:
    #
    #     tgsmt_mean = np.dot(tsematic, tsematic)
    #     y_tmp = tsematic - tgsmt_mean
    #     y_tmp_sigma = np.sqrt(np.dot(y_tmp, y_tmp))
    #     r_ = [y_tmp, y_tmp_sigma]
    #     x1 = np.sqrt(np.dot(x_tmp, x_tmp))
    #     vec_last = np.dot(x_tmp, r_[0]) / (x1 * r_[1]) if (x1 != 0 and r_[1] != 0) else (1 if x1 == r_[1] else -1)
    #     vec_best = vec_best_1 = vec_last
    # # print(x_tmp, r_[0], vec_best, vec_last)

    assert (not (np.isnan(k0) or np.isinf(k0)))

    JX0 = tgdrvt_origin * candidate_1
    JX1 = tgdrvt_origin * candidate_2
    JX_s = np.array([JX0, JX1])
    JXTJX_s = np.dot(JX_s, np.transpose(JX_s))
    u0 = np.max(JXTJX_s) * 10 ** -3
    if u0 == 0:
        u0 = rd.uniform(0, 1)
    res_JT = np.dot(JX_s, vec * tgdrvt_origin).astype(np.float64)
    while (count < time):
        try:
            delta_ks = np.linalg.solve((JXTJX_s + u0 * np.ones(shape=(2, 2))).astype(np.float64),
                                       res_JT)  # 0.1 * np.dot(JX_s, vec * tgdrvt_origin)
        except np.linalg.LinAlgError:
            delta_ks = np.linalg.pinv((JXTJX_s + u0 * np.ones(shape=(2, 2))).astype(np.float64)) @ res_JT
        if not (np.isnan(delta_ks[0]) or np.isnan(delta_ks[1]) or np.isinf(delta_ks[0]) or np.isinf(delta_ks[1])):

            # return (float(k0_best), float(k1_best), float(np.sqrt(vec_best)))
            k0 = k0 - delta_ks[0]
            k1 = k1 - delta_ks[1]

            cdd = k0 * candidate_1 + k1 * candidate_2

            # ===========================1
            vec = np.subtract(cdd, tsematic)

            vec_now = np.dot(vec * tgdrvt, vec)

            # #===========================2
            # x_tmp = cdd - np.sqrt(np.dot(cdd, cdd))
            # x1 = np.sqrt(np.dot(x_tmp, x_tmp))
            # vec_now = np.dot(x_tmp, r_[0]) / (x1 * r_[1]) if (x1 != 0 and r_[1] != 0) else (1 if x1 == r_[1] else -1)
        else:
            vec_now = vec_last
        if vec_now > vec_last:  # ! [ ]
            u0 *= 2
            k0 = k0_best
            k1 = k1_best
        else:
            res_JT = np.dot(JX_s, vec * tgdrvt_origin).astype(np.float64)
            u0 /= 3
            if vec_now < vec_best and not (np.isinf(k0) or np.isinf(k1) or np.isnan(k0) or np.isnan(k1)):
                k0_best = k0
                k1_best = k1
                vec_best = vec_now

        vec_last = vec_now
        count += 1
    assert (not (np.isnan(k0_best) or np.isnan(k1_best) or np.isinf(k0_best) or np.isinf(k1_best)))
    # print(k0_best, k1_best, np.sqrt(vec_best), np.sqrt(vec_best_1))

    # =============================1
    # return [float(k0_best), float(k1_best), float(np.sqrt(vec_best)), float(np.sqrt(vec_best_1))]
    # =============================1
    return [float(k0_best), float(k1_best), float(vec_best), float(vec_best_1)]


def least_square_method(tsematic, candidate_1, candidate_2, tgdrvt):
    numerator = np.dot((candidate_2 - candidate_1) * tgdrvt, tsematic - candidate_1)
    denominator = np.dot((candidate_1 - candidate_2) * tgdrvt, candidate_1 - candidate_2)
    if denominator == 0.:
        return 0
    if np.isinf(numerator).any() or np.isinf(denominator) or np.isnan(denominator).any() or np.isnan(numerator):
        return 0
    if (math.isnan(numerator / denominator)):
        raise ValueError("why here..", numerator, denominator, candidate_1, candidate_2, tsematic)
    return numerator / denominator


def effect_test(tsematic, origin, candidate_1, candidate_2, k, tgdrvt, cdd_size, serious=False, mask=None):
    cdd = k[0] * candidate_1 + k[1] * candidate_2
    vec = np.subtract(tsematic, cdd)

    # tgdrvt_1 = tgdrvt * mask
    # tgdrvt_2 = tgdrvt * ((mask + 1) % 2)
    # effect_1 = np.sqrt(np.dot(vec * tgdrvt_1, vec))
    # effect_2 = np.sqrt(np.dot(vec * tgdrvt_2, vec))

    effect = np.sqrt(np.dot(vec * tgdrvt, vec) / len(vec))

    # effect_1 = np.sqrt(np.dot(vec, vec))
    # vec = np.subtract(tsematic, candidate_1) * tgdrvt
    # effect_1 = np.sqrt(np.dot(vec, vec))
    # vec = np.subtract(tsematic, candidate_2) * tgdrvt
    # effect_2 = np.sqrt(np.dot(vec, vec))

    vec = np.subtract(tsematic, origin)
    origin_effect = np.sqrt(np.dot(vec * tgdrvt, vec) / len(vec))
    # origin_effect_1 = np.sqrt(np.dot(vec * tgdrvt_1, vec))
    # origin_effect_2 = np.sqrt(np.dot(vec * tgdrvt_2, vec))

    # origin_effect_1 = np.sqrt(np.dot(vec, vec))
    # if effect_1 < effect:
    #     k = 0
    #     effect = effect_1
    # if effect_2 < effect:
    #     k = 1
    #     effect = effect_2
    # vec_1 = float(cdd_size[0] + cdd_size[1] + 3) - float(cdd_size[2])
    crm_size = cdd_size[3] - cdd_size[2]

    if serious:
        return ((effect != 0 and (0.9999 ** (float(cdd_size[0] + cdd_size[1] + 3 + crm_size))) / effect > (0.9999 ** (
            float(cdd_size[3]))) / origin_effect) * 0.99, k, effect,
                origin_effect)  # , origin_effect_1, origin_effect_2, effect, effect - origin_effect, origin_effect)
    else:
        return (effect - origin_effect < origin_effect * 0.0, k, effect,
                origin_effect)  # , origin_effect_1, origin_effect_2, effect, effect - origin_effect, origin_effect)


def bounds_check(subtr: TreeNode, smt_rg, k, data_rg):
    rg = subtr.getRange(data_rg)
    ancestors = subtr.getAncestors()
    for x in ancestors:
        child = x[0].getChilds()
        if ((x[0].nodeval.name == '/'and child[0].print_exp_subtree() != child[1].print_exp_subtree() and x[1] == 1)) and (
                rg[0] <= 0. <= rg[1] or math.fabs(rg[0]) == 0.0 or math.fabs(rg[1]) == 0.0) :
            return False
        if x[0].nodeval.name == '/' and x[1] == 0:
            child_rg_1 = child[1].getRange(data_rg)
            if (child_rg_1[0] <= 0. <= child_rg_1[1] or math.fabs(child_rg_1[0]) == 0.0 or math.fabs(
                    child_rg_1[1]) == 0.0) and child[0].print_exp_subtree() != child[
                1].print_exp_subtree():
                return False
        trs = x[0].getChilds().copy()
        trs.pop(x[1])

        rgs = [node.getRange(data_rg) for node in trs]
        rgs.insert(x[1], rg)
        rg = PyGP.rg_compute(x[0], 0, rgs)
    return True


def roth(rd, tgs, res_vals):
    tg = 0
    if len(tgs) > 0 and (len(res_vals) == 0):  # or rd.uniform(0, 1) < 0):
        tg = tgs[rd.integers(0, len(tgs))]
    else:
        
        max_res_vals = np.max(res_vals)
        res_vals = np.array([math.e ** (res_vals[i] - max_res_vals) for i in range(len(res_vals))])
        res_vals /= np.sum(res_vals)
        tg_randval = rd.uniform(0, 1)
        res = 0
        for i in range(len(res_vals)):
            res += res_vals[i]
            if res > tg_randval:
                tg = tgs[i]
                break
    return tg


import dill


def _crossover(rd, pprogs: [Program], smts: PopSemantic, funcs, depth_limit):
    # crsover_time = [0, 0, 0]
    progs = []
    idx = 0
    idx_suc = 0
    prog_depth_max = 0
    data_rg = smts.get_datarg()
    idx_best = 0

    num = 0
    for i in range(len(pprogs)):

        indiv1: Program = pprogs[i]
        child = indiv1
        num += 1
        idx += 1
        id = indiv1.prog_id
        indiv1.seman_sign = -1

        res_tgs = smts.compute_tg(id)
        tg_idx = roth(rd, *res_tgs)

        rlt_posi = smts.get_tgnode_posi(id, tg_idx)
        subtree3 = child.getSubTree(rlt_posi)
        s3_height, s3_size, s3_rlt_depth = subtree3.height(), subtree3.inner_size, subtree3.relative_depth()
        h_limit = PyGP.DEPTH_MAX_SIZE - subtree3.relative_depth() - 1
        h_init = 0
        h_limit = 1 if h_limit <= 0 else h_limit

        tgdrvt_origin = smts.get_drvt_d(id, tg_idx)
        smt_size, h_rg = smts.get_smt_size(h_limit, init_height=h_init)

        real_nums = PyGP.SEMANTIC_NUM if smt_size > PyGP.SEMANTIC_NUM else smt_size
        r_idxs = rd.choice(range(smt_size), real_nums, replace=False)

        tgsmt = smts.get_tgsmt_d(id, tg_idx)
        # assert (len(r_idxs) > 0)
        if len(r_idxs) > 1 and \
                not (any(np.isnan(tgdrvt_origin)) or any(np.isinf(tgdrvt_origin)) or any(np.isnan(tgsmt)) or any(
                    np.isinf(tgsmt))):
            t0 = time.time()

            # rlt_posi = smts.get_tgnode_posi(id, tg_idx)
            tr_origin = subtree3  # child.getSubTree(rlt_posi)
            cdd_origin = smts.get_snode_tgsmt(id, tg_idx)

            tgdrvt = np.fabs(tgdrvt_origin)
            tgdrvt_test = PyGP.abs_normalize(tgdrvt)

            depth_new = depth_limit 
            mask = np.ones(len(tgsmt)) 
            candidates = smts.get_smt_trs(h_rg, r_idxs)
            cdd_c = np.ones(len(tgsmt))
            candidates.append(TR(expr='1', smt= cdd_c, isize=0, rg=[1., 1.], tr=TreeNode(1.)))

            (indiv_idx, indivs, _, k, succeed) = indivSelect_sem_5(tsematic_=tgsmt, candidates=candidates,
                                                                tgdrvt_=tgdrvt_test,
                                                                tgdrvt_origin_=tgdrvt_origin,
                                                                candidate_origin_=cdd_origin,
                                                                mask=mask, depth_limit=depth_new,
                                                                s3_size=(s3_size, child.root.inner_size),
                                                                tr_origin=tr_origin, org=PyGP.DEPTH_MAX_SIZE - (
                            s3_rlt_depth + s3_height) >= 1 and s3_height > 1)

            if isinstance(indiv_idx, list):

                trs_cdd = [candidates[indiv_idx[0]].tree, candidates[indiv_idx[1]].tree]
                if succeed:
                    indivs = [candidates[indiv_idx[0]].semantic, candidates[indiv_idx[1]].semantic]
                    effect_better = [True]  # effect_test(tgsmt, cdd_origin,
                else:
                    effect_better = [False]

            else:

                trs_cdd = candidates[indiv_idx].tree
                effect_better = [True] if succeed else [False]
            if effect_better[0]:
                if effect_better[0]:
                    idx_suc += 1
                if not isinstance(trs_cdd, list):
                    subtree1: TreeNode = trs_cdd
                    if not (math.fabs(k - 1) == 0.0):
                        tr1 = TreeNode(funcs.funcSelect_n('mul'))
                        tr1.setChilds([subtree1, TreeNode(k, parent=(tr1, 1))])
                        subtree1.setParent((tr1, 0))
                        tr3 = tr1
                    else:
                        tr3 = subtree1
                else:
                    subtree1: TreeNode = trs_cdd[0]
                    subtree2: TreeNode = trs_cdd[1]

                    if (math.fabs(k[1]) == 0.0):
                        if subtree1.dtype == 'Const':
                            tr1 = TreeNode(subtree1.nodeval * k[0])
                        else:
                            tr1 = TreeNode(funcs.funcSelect_n('mul'))
                            tr1.setChilds([subtree1, TreeNode(k[0], parent=(tr1, 1))])
                            subtree1.setParent((tr1, 0))
                        tr3 = tr1
                    elif (math.fabs(k[0]) == 0.0):
                        if subtree2.dtype == 'Const':
                            tr2 = TreeNode(subtree2.nodeval * k[1])
                        else:
                            tr2 = TreeNode(funcs.funcSelect_n('mul'))
                            tr2.setChilds([subtree2, TreeNode(k[1], parent=(tr2, 1))])
                            subtree2.setParent((tr2, 0))
                        tr3 = tr2
                    else:
                        if subtree1.dtype == 'Const':
                            tr1 = TreeNode(subtree1.nodeval * k[0])
                        else:
                            tr1 = TreeNode(funcs.funcSelect_n('mul'))
                            tr1.setChilds([subtree1, TreeNode(k[0], parent=(tr1, 1))])
                            subtree1.setParent((tr1, 0))
                        if subtree2.dtype == 'Const':
                            tr2 = TreeNode(subtree2.nodeval * k[1])
                        else:
                            tr2 = TreeNode(funcs.funcSelect_n('mul'))
                            tr2.setChilds([subtree2, TreeNode(k[1], parent=(tr2, 1))])
                            subtree2.setParent((tr2, 0))
                        if tr1.dtype == 'Const' and tr2.dtype == 'Const':
                            tr3 = TreeNode(tr1.nodeval + tr2.nodeval)
                        else:
                            tr3 = TreeNode(funcs.funcSelect_n('add'))
                            tr3.setChilds([tr1, tr2])
                            tr1.setParent((tr3, 0))
                            tr2.setParent((tr3, 1))

                if subtree3.parent is not None:
                    tr3.setParent(subtree3.parent)
                else:
                    child.root = tr3

                if isinstance(trs_cdd, list):
                    rg0, rg1 = candidates[indiv_idx[0]].range, candidates[indiv_idx[1]].range
                else:
                    rg0 = candidates[indiv_idx].range

                if PyGP.INTERVAL_COMPUTE and (
                        (isinstance(trs_cdd, list) and not bounds_check(tr3, (rg0, rg1), k, data_rg))
                        or (not isinstance(trs_cdd, list) and not bounds_check(tr3, rg0, None, data_rg))):
                    if tr3.parent is not None:
                        subtree3.setParent(tr3.parent)
                    else:
                        child.root = subtree3

                child.sizeUpdate()
                progs.append(child)
                if prog_depth_max < child.depth:
                    prog_depth_max = child.depth
            else:
                progs.append(None)
            # crsover_time[2] += t1 - t0
            t1 = time.time()
        else:
            progs.append(None)
    num += 1
    return progs


from .base import BaseCrossover


class SMT_Weight_Crossover_LV2(BaseCrossover):
    def __init__(self, pop_size):
        self.pop_size = pop_size

    def run(self, pprogs, funcs, depth_limit=10):
        smts = self.semantics
        rd = self.rg
        return _crossover(rd, pprogs, smts, funcs, depth_limit)