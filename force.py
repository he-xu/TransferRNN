def ternarize(w_new, cam_num):
    w_order = np.argsort(np.abs(w_new.T), axis=0)
    w_sorted = w_new.T[w_order, np.arange(w_order.shape[1])]
    w_sorted[:-cam_num, :]=0
    w_order_order = np.argsort(w_order, axis=0)
    w_undone = w_sorted[w_order_order, np.arange(w_order_order.shape[1])].T
    w_undone[w_undone>0] = 1
    w_undone[w_undone<0] = -1
    return w_undone

def update_weight(rate_psc, rate_teacher, w_real, w_ternary, n=6, m=1, cam_num=63, learning_rate=0.1):
    rate_recurrent = w_ternary.dot(rate_psc)
    rate_teacher_tile = np.tile(rate_teacher, (n,m))
    error = rate_recurrent - rate_teacher_tile
    d_w = 0
    for t in range(num_timesteps):
        r_t = rate_psc[:, t][:,np.newaxis]
        P_up = P_prev.dot(r_t.dot(r_t.T.dot(P_prev)))
        P_down = 1 + r_t.T.dot(P_prev.dot(r_t))
        P_t =  P_prev - P_up / P_down
        e_t = error[:, t][:,np.newaxis]
        d_w += e_t.dot(r_t.T.dot(P_t))
    d_w = d_w / num_timesteps
    w_new = w_ternary - learning_rate*d_w
    w_ternary = ternarize(w_new, cam_num)
    norm_ratio = np.linalg.norm(w_new, 'fro')/np.linalg.norm(w_ternary, 'fro')
    if norm_ratio > 1:
        c_grad = 1
    else:
        c_grad = -1
    return w_ternary, c_grad

    
