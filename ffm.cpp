#pragma GCC diagnostic ignored "-Wunused-result" 
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <new>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <cstring>
#include <vector>
#include <pmmintrin.h>

#if defined USEOMP
#include <omp.h>
#endif

#include "ffm.h"

namespace ffm {

namespace {

using namespace std;

ffm_int const kALIGNByte = 16;
ffm_int const kALIGN = kALIGNByte/sizeof(ffm_float);
ffm_int const kCHUNK_SIZE = 2000000000;
ffm_int const kMaxLineSize = 100000;

inline float qrsqrt(float x)
{
    _mm_store_ss(&x, _mm_rsqrt_ps(_mm_load1_ps(&x)));
    return x;
}

ffm_int const kBIN_SIZE = (1 << 24) - 1;

inline ffm_int hash_idx(ffm_int a, ffm_int b)
{
    return (((((ffm_long)a+b)*(a+b+1)>>1)+b) % kBIN_SIZE);
}

ffm_float calc_w_norm2(ffm_model &model)
{
    ffm_float w_norm2 = 0.0;
    ffm_float *ptr = model.W;
    for(ffm_int j = 0; j < model.n; j++)
    {
        for(ffm_int f = 0; f < model.m; f++)
        {
            for(ffm_int d = 0; d < model.k; d++, ptr++)
                w_norm2 += (*ptr)*(*ptr);
        }
    }
    return sqrt(w_norm2);
}

inline ffm_float wTx(
    ffm_node *begin,
    ffm_node *end,
    ffm_float r,
    ffm_model &model, 
    ffm_float kappa=0, 
    ffm_float eta=0, 
    ffm_float lambda=0, 
    bool do_update=false)
{
    ffm_long align0 = (ffm_long)model.k*2;
    ffm_long align1 = (ffm_long)model.m*align0;
    ffm_float sqrt_r = sqrt(r);

    __m128 XMMkappa = _mm_set1_ps(kappa);
    __m128 XMMeta = _mm_set1_ps(eta);
    __m128 XMMlambda = _mm_set1_ps(lambda);

    __m128 XMMt = _mm_setzero_ps();

    ffm_float t = 0, linear = 0, poly2 = 0;
    for(ffm_node *N1 = begin; N1 != end; N1++)
    {
        ffm_int j1 = N1->j;
        ffm_int f1 = N1->f;
        ffm_float v1 = N1->v;
        if(j1 >= model.n || f1 >= model.m || !model.tr_feat_idx.count(j1))
            continue;

        ffm_float &wl = model.WL[j1 * 2];
        if(do_update)
        {
            ffm_float &wlg = model.WL[j1 * 2 + 1 ];
            ffm_float g = lambda * wl + kappa * v1 * sqrt_r;
            wlg += g * g;
            wl -= eta * qrsqrt(wlg) * g;
        }

        else
        {
            t += wl * v1 * sqrt_r;
            linear += wl * v1 * sqrt_r;
        }

        for(ffm_node *N2 = N1+1; N2 != end; N2++)
        {
            ffm_int j2 = N2->j;
            ffm_int f2 = N2->f;
            ffm_float v2 = N2->v;
            if(j2 >= model.n || f2 >= model.m || !model.tr_feat_idx.count(j2) || f1 == f2)
                continue;

            ffm_float *w1 = model.W + j1*align1 + f2*align0;
            ffm_float *w2 = model.W + j2*align1 + f1*align0;

            ffm_int hidx = hash_idx(j1, j2);
            ffm_float &wp = model.WP[2*hidx];
            ffm_float pair_v = v1*v2*r;

            if(do_update)
            {
                ffm_float &wpg = model.WP[2*hidx+1];
                ffm_float g = lambda*wl + kappa * pair_v;
                wpg += g * g;
                wp -= eta * qrsqrt(wpg) * g;
            }
            else
            {
                t+= wp*pair_v;
                poly2 += wp*pair_v;
            }

            __m128 XMMv = _mm_set1_ps(pair_v);

            if(do_update)
            {
                __m128 XMMkappav = _mm_mul_ps(XMMkappa, XMMv);

                ffm_float *wg1 = w1 + model.k;
                ffm_float *wg2 = w2 + model.k;
                for(ffm_int d = 0; d < model.k; d += 4)
                {
                    __m128 XMMw1 = _mm_load_ps(w1+d);
                    __m128 XMMw2 = _mm_load_ps(w2+d);

                    __m128 XMMwg1 = _mm_load_ps(wg1+d);
                    __m128 XMMwg2 = _mm_load_ps(wg2+d);

                    __m128 XMMg1 = _mm_add_ps(
                                   _mm_mul_ps(XMMlambda, XMMw1),
                                   _mm_mul_ps(XMMkappav, XMMw2));
                    __m128 XMMg2 = _mm_add_ps(
                                   _mm_mul_ps(XMMlambda, XMMw2),
                                   _mm_mul_ps(XMMkappav, XMMw1));

                    XMMwg1 = _mm_add_ps(XMMwg1, _mm_mul_ps(XMMg1, XMMg1));
                    XMMwg2 = _mm_add_ps(XMMwg2, _mm_mul_ps(XMMg2, XMMg2));

                    XMMw1 = _mm_sub_ps(XMMw1, _mm_mul_ps(XMMeta, 
                            _mm_mul_ps(_mm_rsqrt_ps(XMMwg1), XMMg1)));
                    XMMw2 = _mm_sub_ps(XMMw2, _mm_mul_ps(XMMeta, 
                            _mm_mul_ps(_mm_rsqrt_ps(XMMwg2), XMMg2)));

                    _mm_store_ps(w1+d, XMMw1);
                    _mm_store_ps(w2+d, XMMw2);

                    _mm_store_ps(wg1+d, XMMwg1);
                    _mm_store_ps(wg2+d, XMMwg2);
                }
            }
            else
            {
                for(ffm_int d = 0; d < model.k; d += 4)
                {
                    __m128  XMMw1 = _mm_load_ps(w1+d);
                    __m128  XMMw2 = _mm_load_ps(w2+d);

                    XMMt = _mm_add_ps(XMMt, 
                           _mm_mul_ps(_mm_mul_ps(XMMw1, XMMw2), XMMv));
                }
            }
        }
    }

    ffm_float &wb = model.WB[0];
    if(do_update)
    {
        ffm_float &wbg = model.WB[1];
        ffm_float g = kappa;
        wbg += g * g;
        wb -= eta * qrsqrt(wbg) * g;
    }
    else
    {
        t += wb;
    }

    if(do_update)
        return 0;

    XMMt = _mm_hadd_ps(XMMt, XMMt);
    XMMt = _mm_hadd_ps(XMMt, XMMt);
    ffm_float t_sse;
    _mm_store_ss(&t_sse, XMMt);

    //cout << "linear: " << t << " ffm: " << t_sse << endl;

    return t + t_sse;

    return t;
}




ffm_float* malloc_aligned_float(ffm_long size)
{
    void *ptr;

#ifdef _WIN32
    ptr = _aligned_malloc(size*sizeof(ffm_float), kALIGNByte);
    if(ptr == nullptr)
        throw bad_alloc();
#else
    int status = posix_memalign(&ptr, kALIGNByte, size*sizeof(ffm_float));
    if(status != 0)
        throw bad_alloc();
#endif
    
    return (ffm_float*)ptr;
}

ffm_model* init_model(ffm_int n, ffm_int m, ffm_parameter param)
{
    ffm_int k_aligned = (ffm_int)ceil((ffm_double)param.k/kALIGN)*kALIGN;

    ffm_model *model = new ffm_model;
    model->n = n;
    model->k = k_aligned;
    model->m = m;
    model->W = nullptr;
    model->WL = nullptr;
    model->WB = nullptr;
    model->WP = nullptr;
    model->normalization = param.normalization;
    
    try
    {
        model->W = malloc_aligned_float((ffm_long)n*m*k_aligned*2);
        model->WL = malloc_aligned_float(n*2);
        model->WB = malloc_aligned_float(2);
        model->WP = malloc_aligned_float(2*kBIN_SIZE);
    }
    catch(bad_alloc const &e)
    {
        ffm_destroy_model(&model);
        throw;
    }

    ffm_float coef = 1.0f/sqrt(param.k);
    ffm_float *w = model->W;

    default_random_engine generator;
    uniform_real_distribution<ffm_float> distribution(0.0, 1.0);

    for(ffm_int j = 0; j < model->n; j++)
    {
        for(ffm_int f = 0; f < model->m; f++)
        {
            for(ffm_int d = 0; d < param.k; d++, w++)
                *w = coef*distribution(generator);
            for(ffm_int d = param.k; d < k_aligned; d++, w++)
                *w = 0;
            for(ffm_int d = k_aligned; d < 2*k_aligned; d++, w++)
                *w = 1;
        }
    }

    for(ffm_int j = 0; j < model->n; j++)
    {
        model->WL[j * 2] = 0;
        model->WL[j * 2 + 1] = 1;
    }

    for (ffm_int j2 = 0; j2 < kBIN_SIZE; j2++) {
        model->WP[2*j2] = 0;
        model->WP[2*j2+1] = 1;
    }

    model->WB[0] = 0;
    model->WB[1] = 1;

    return model;
}


vector<ffm_float> normalize(ffm_problem &prob)
{
    vector<ffm_float> R(prob.l);
#if defined USEOMP
#pragma omp parallel for schedule(static)
#endif
    for(ffm_int i = 0; i < prob.l; i++)
    {
        ffm_float norm = 0;
        for(ffm_long p = prob.P[i]; p < prob.P[i+1]; p++)
            norm += prob.X[p].v*prob.X[p].v;
        R[i] = 1/norm;
    }

    return R;
}

shared_ptr<ffm_model> train(
    ffm_problem *tr, 
    vector<ffm_int> &order, 
    ffm_parameter param, 
    ffm_model *raw_model,
    ffm_problem *va=nullptr)
{
#if defined USEOMP
    ffm_int old_nr_threads = omp_get_num_threads();
    omp_set_num_threads(param.nr_threads);
#endif

    if (raw_model == nullptr) {
        raw_model = init_model(tr->n, tr->m, param);
    }
    shared_ptr<ffm_model> model = 
        shared_ptr<ffm_model>(raw_model,
            [] (ffm_model *ptr) { ffm_destroy_model(&ptr); });

    ffm_int nr_pos = 0;
    for(ffm_int i = 0; i < tr->l; i++)
    {
        
        ffm_float y = tr->Y[i];
        
        if (y > 0)
            nr_pos++;

        ffm_node *begin = &tr->X[tr->P[i]];

        ffm_node *end = &tr->X[tr->P[i+1]];

        for (ffm_node *N = begin; N != end; N++)
        {
            ffm_int j = N->j;
            model->tr_feat_idx.insert(j);
        }
    }
    model->tr_avg = ffm_float(nr_pos)/tr->l;
    model->tr_avg = -log(1/model->tr_avg-1);
    cout <<"Global bias:" <<model->tr_avg << endl;

    vector<ffm_float> R_tr, R_va;
    if(param.normalization)
    {
        R_tr = normalize(*tr);
        if(va != nullptr)
            R_va = normalize(*va);
    }
    else
    {
        R_tr = vector<ffm_float>(tr->l, 1);
        if(va != nullptr)
            R_va = vector<ffm_float>(va->l, 1);
    }

    bool auto_stop = param.auto_stop && va != nullptr && va->l != 0;

    ffm_int k_aligned = (ffm_int)ceil((ffm_double)param.k/kALIGN)*kALIGN;
    ffm_long w_size = (ffm_long)model->n * model->m * k_aligned * 2;
    vector<ffm_float> prev_W;
    if(auto_stop)
        prev_W.assign(w_size, 0);
    ffm_double best_auc_loss = numeric_limits<ffm_double>::max();

    if(!param.quiet)
    {
        if(param.auto_stop && (va == nullptr || va->l == 0))
            cerr << "warning: ignoring auto-stop because there is no validation set" << endl;

        cout.width(4);
        cout << "iter";
        cout.width(13);
        cout << "tr_logloss";
        if(va != nullptr && va->l != 0)
        {
            cout.width(13);
            cout << "va_aucloss";
        }
        cout.width(13);
        cout << "|W|";
        cout << endl;
    }

    for(ffm_int iter = 1; iter <= param.nr_iters; iter++)
    {
        ffm_double tr_loss = 0;
        if(param.random)
            random_shuffle(order.begin(), order.end());
#if defined USEOMP
#pragma omp parallel for schedule(static) reduction(+: tr_loss)
#endif
        for(ffm_int ii = 0; ii < (ffm_int)order.size(); ii++)
        {
            ffm_int i = order[ii];

            ffm_float y = tr->Y[i];
            
            ffm_node *begin = &tr->X[tr->P[i]];

            ffm_node *end = &tr->X[tr->P[i+1]];

            ffm_float r = R_tr[i];

            ffm_float t = wTx(begin, end, r, *model);

            ffm_float expnyt = exp(-y*t);

            tr_loss += log(1+expnyt);
               
            ffm_float kappa = -y*expnyt/(1+expnyt);

            wTx(begin, end, r, *model, kappa, param.eta, param.lambda, true);


        }
        ffm_float w_norm2 = calc_w_norm2(*model);

        if(!param.quiet)
        {
            tr_loss /= tr->l;

            cout.width(4);
            cout << iter;
            cout.width(13);
            cout << fixed << setprecision(5) << tr_loss;

            if(va != nullptr && va->l != 0)
            {
                vector<ffm_float> va_scores(va->l, 0);
                vector<ffm_int> va_orders(va->l, 0);
#pragma omp parallel for schedule(static)
                for(ffm_long i = 0; i < va->l; i++)
                {
                    ffm_node *begin = &va->X[va->P[i]];

                    ffm_node *end = &va->X[va->P[i+1]];

                    ffm_int nr_nnz = 0;
                    ffm_int nr_missed = 0;
                    for (ffm_node *N = begin; N != end; N++)
                    {
                        nr_nnz++;
                        ffm_int j = N->j;
                        if (!model->tr_feat_idx.count(j))
                            nr_missed++;
                    }

                    ffm_float missed_bias = 0.0;

                    if (!nr_missed)
                        missed_bias = 0.0;
                    else
                        missed_bias = model->tr_avg*(nr_missed*(nr_missed-1))/(nr_nnz*(nr_nnz-1));

                    begin = &va->X[va->P[i]];

                    end = &va->X[va->P[i+1]];

                    ffm_float r = R_va[i];

                    ffm_float t = wTx(begin, end, r, *model);

                    t += missed_bias;

                    va_scores[i] = t;
                    va_orders[i] = i;
                }
                sort(va_orders.begin(), va_orders.end(), [&va_scores, &va] (ffm_int i, ffm_int j) {return va_scores[i] < va_scores[j];});
                ffm_float prev_score = va_scores[0];
                ffm_long M = 0, N = 0;
                ffm_long begin=0, stuck_pos=0, stuck_neg=0;
                ffm_float sum_pos_rank = 0;
                for (ffm_long i = 0; i < va->l; i++)
                {
                    ffm_int sorted_i = va_orders[i];

                    ffm_float score = va_scores[sorted_i];

                    if (score != prev_score)
                    {
                        sum_pos_rank += stuck_pos*(begin+begin-1+stuck_pos+stuck_neg)*0.5;
                        prev_score = score;
                        begin = i;
                        stuck_neg = 0;
                        stuck_pos = 0;
                    }

                    ffm_float label = va->Y[sorted_i];

                    if (label > 0)
                    {
                        M++;
                        stuck_pos ++;
                    }
                    else
                    {
                        N++;
                        stuck_neg ++;
                    }
                }
                sum_pos_rank += stuck_pos*(begin+begin-1+stuck_pos+stuck_neg)*0.5;
                ffm_float auc_loss = (sum_pos_rank - 0.5*M*(M+1)) / (M*N);

                cout.width(13);
                cout << fixed << setprecision(5) << auc_loss;

                if(auto_stop)
                {
                    if(auc_loss > best_auc_loss)
                    {
                        memcpy(model->W, prev_W.data(), w_size*sizeof(ffm_float));
                        cout << endl << "Auto-stop. Use model at " << iter-1 << "th iteration." << endl;
                        break;
                    }
                    else
                    {
                        memcpy(prev_W.data(), model->W, w_size*sizeof(ffm_float));
                        best_auc_loss = auc_loss; 
                    }
                }
            }
	    cout.width(13);
	    cout << fixed << setprecision(5) << w_norm2;
            cout << endl;
        }
    }

    //shrink_model(*model, param.k);

#if defined USEOMP
    omp_set_num_threads(old_nr_threads);
#endif

    return model;
}

// TODO: This function will be merged with train().
shared_ptr<ffm_model> train_on_disk(
    string tr_path,
    string va_path,
    ffm_parameter param,
    ffm_model *raw_model)
{
#if defined USEOMP
    ffm_int old_nr_threads = omp_get_num_threads();
    omp_set_num_threads(param.nr_threads);
#endif

    FILE *f_tr = fopen(tr_path.c_str(), "rb");
    FILE *f_va = nullptr;
    if(!va_path.empty())
        f_va = fopen(va_path.c_str(), "rb");

    ffm_int m, n, max_l;
    ffm_long max_nnz;
    fread(&m, sizeof(ffm_int), 1, f_tr);
    fread(&n, sizeof(ffm_int), 1, f_tr);
    fread(&max_l, sizeof(ffm_int), 1, f_tr);
    fread(&max_nnz, sizeof(ffm_long), 1, f_tr);

    if (raw_model == nullptr)
        raw_model = init_model(n, m, param);

    shared_ptr<ffm_model> model = 
        shared_ptr<ffm_model>(raw_model,
            [] (ffm_model *ptr) { ffm_destroy_model(&ptr); });

    vector<ffm_float> Y;
    Y.reserve(max_l);
    vector<ffm_float> R;
    R.reserve(max_l);
    vector<ffm_long> P;
    P.reserve(max_l+1);
    vector<ffm_node> X;
    X.reserve(max_nnz);

    fseek(f_tr, 3*sizeof(ffm_int)+sizeof(ffm_long), SEEK_SET);

    ffm_int tr_l = 0, nr_pos=0;
    while(true)
    {
        ffm_int l;
        fread(&l, sizeof(ffm_int), 1, f_tr);
        tr_l += l;
        if(l == 0)
            break;

        Y.resize(l);
        fread(Y.data(), sizeof(ffm_float), l, f_tr);

        R.resize(l);
        fread(R.data(), sizeof(ffm_float), l, f_tr);

        P.resize(l+1);
        fread(P.data(), sizeof(ffm_long), l+1, f_tr);

        X.resize(P[l]);
        fread(X.data(), sizeof(ffm_node), P[l], f_tr);

        for(ffm_int i = 0; i < l; i++)
        {
            
            ffm_float y = Y[i];
            
            if (y > 0)
                nr_pos++;

            ffm_node *begin = &X[P[i]];

            ffm_node *end = &X[P[i+1]];

            for (ffm_node *N = begin; N != end; N++)
            {
                ffm_int j = N->j;
                model->tr_feat_idx.insert(j);
            }
        }
    }
    model->tr_avg = ffm_float(nr_pos)/tr_l;
    model->tr_avg = -log(1/model->tr_avg-1);
    cout << "Global bias:" << model->tr_avg << endl;

    bool auto_stop = param.auto_stop && !va_path.empty();

    ffm_int k_aligned = (ffm_int)ceil((ffm_double)param.k/kALIGN)*kALIGN;
    ffm_long w_size = (ffm_long)model->n * model->m * k_aligned * 2;
    vector<ffm_float> prev_W;
    if(auto_stop)
        prev_W.assign(w_size, 0);
    ffm_double best_auc_loss = numeric_limits<ffm_double>::max();

    if(!param.quiet)
    {
        if(param.auto_stop && va_path.empty())
            cerr << "warning: ignoring auto-stop because there is no validation set" << endl;
        cout.width(4);
        cout << "iter";
        cout.width(13);
        cout << "tr_logloss";
        if(!va_path.empty())
        {
            cout.width(13);
            cout << "va_aucloss";
            cout.width(13);
            cout << "va_logloss";
        }
        cout << endl;
    }

    for(ffm_int iter = 1; iter <= param.nr_iters; iter++)
    {
        ffm_double tr_loss = 0;
        ffm_double te_loss = 0;

        fseek(f_tr, 3*sizeof(ffm_int)+sizeof(ffm_long), SEEK_SET);

        ffm_int tr_l = 0;
        while(true)
        {
            ffm_int l;
            fread(&l, sizeof(ffm_int), 1, f_tr);
            tr_l += l;
            if(l == 0)
                break;

            Y.resize(l);
            fread(Y.data(), sizeof(ffm_float), l, f_tr);

            R.resize(l);
            fread(R.data(), sizeof(ffm_float), l, f_tr);

            P.resize(l+1);
            fread(P.data(), sizeof(ffm_long), l+1, f_tr);

            X.resize(P[l]);
            fread(X.data(), sizeof(ffm_node), P[l], f_tr);
            vector<ffm_int> orders(l);
            iota(orders.begin(), orders.end(), 0);
            if(param.random)
                random_shuffle(orders.begin(), orders.end());

#if defined USEOMP
#pragma omp parallel for schedule(static) reduction(+: tr_loss)
#endif
            for(ffm_int i_ = 0; i_ < l; i_++)
            {
                ffm_int i = orders[i_];
                ffm_float y = Y[i];
                
                ffm_node *begin = &X[P[i]];

                ffm_node *end = &X[P[i+1]];

                ffm_float r = param.normalization? R[i] : 1;

                ffm_float t = wTx(begin, end, r, *model);

                ffm_float expnyt = exp(-y*t);

                tr_loss += log(1+expnyt);
                   
                ffm_float kappa = -y*expnyt/(1+expnyt);

                wTx(begin, end, r, *model, kappa, param.eta, param.lambda, true);
            }
        }

        if(!param.quiet)
        {
            tr_loss /= tr_l;

            cout.width(4);
            cout << iter;
            cout.width(13);
            cout << fixed << setprecision(5) << tr_loss;

            if(f_va != nullptr)
            {
                fseek(f_va, 3*sizeof(ffm_int)+sizeof(ffm_long), SEEK_SET);

                ffm_int va_l = 0;
                ffm_int base_l = 0;
                ffm_double va_loss = 0;
                vector<ffm_float> va_scores;
                vector<ffm_float> va_Y;
                vector<ffm_int> va_orders;
                while(true)
                {
                    ffm_int l;
                    fread(&l, sizeof(ffm_int), 1, f_va);
                    va_l += l;
                    if(l == 0)
                        break;

                    va_scores.resize(va_l);
                    va_orders.resize(va_l);
                    va_Y.resize(va_l);

                    vector<ffm_float> Y(l);
                    fread(Y.data(), sizeof(ffm_float), l, f_va);

                    vector<ffm_float> R(l);
                    fread(R.data(), sizeof(ffm_float), l, f_va);

                    vector<ffm_long> P(l+1);
                    fread(P.data(), sizeof(ffm_long), l+1, f_va);

                    vector<ffm_node> X(P[l]);
                    fread(X.data(), sizeof(ffm_node), P[l], f_va);

#if defined USEOMP
#pragma omp parallel for schedule(static) reduction(+: te_loss)
#endif
                    for(ffm_int i = 0; i < l; i++)
                    {
                        ffm_float y = Y[i];
                        va_Y[base_l+i] = y;

                        ffm_node *begin = &X[P[i]];

                        ffm_node *end = &X[P[i+1]];

                        ffm_int nr_nnz = 0;
                        ffm_int nr_missed = 0;
                        for (ffm_node *N = begin; N != end; N++)
                        {
                            nr_nnz++;
                            ffm_int j = N->j;
                            if (!model->tr_feat_idx.count(j))
                                nr_missed++;
                        }

                        ffm_float missed_bias = 0.0;

                        if (!nr_missed)
                            missed_bias = 0.0;
                        else
                            missed_bias = model->tr_avg*(nr_missed*(nr_missed-1))/(nr_nnz*(nr_nnz-1));

                        
                        begin = &X[P[i]];

                        end = &X[P[i+1]];

                        ffm_float r = param.normalization? R[i] : 1;

                        ffm_float t = wTx(begin, end, r, *model);

                        t += missed_bias;
			te_loss += log(1+exp(-y*t));

                        va_scores[base_l+i] = t;
                        va_orders[base_l+i] = base_l+i;

                    }
                    base_l += l;
                }

                sort(va_orders.begin(), va_orders.end(), [&va_scores] (ffm_int i, ffm_int j) {return va_scores[i] < va_scores[j];});
                ffm_float prev_score = va_scores[0];
                ffm_long M = 0, N = 0;
                ffm_long begin=0, stuck_pos=0, stuck_neg=0;
                ffm_float sum_pos_rank = 0;
                for (ffm_long i = 0; i < va_l; i++)
                {
                    ffm_int sorted_i = va_orders[i];

                    ffm_float score = va_scores[sorted_i];

                    if (score != prev_score)
                    {
                        sum_pos_rank += stuck_pos*(begin+begin-1+stuck_pos+stuck_neg)*0.5;
                        prev_score = score;
                        begin = i;
                        stuck_neg = 0;
                        stuck_pos = 0;
                    }

                    ffm_float label = va_Y[sorted_i];

                    if (label > 0)
                    {
                        M++;
                        stuck_pos ++;
                    }
                    else
                    {
                        N++;
                        stuck_neg ++;
                    }
                }
                sum_pos_rank += stuck_pos*(begin+begin-1+stuck_pos+stuck_neg)*0.5;
                ffm_float auc_loss = (sum_pos_rank - 0.5*M*(M+1)) / (M*N);


                cout.width(13);
                cout << fixed << setprecision(5) << auc_loss;
                cout.width(13);
                cout << fixed << setprecision(5) << te_loss/va_l;

                if(auto_stop)
                {
                    if(va_loss > best_auc_loss)
                    {
                        memcpy(model->W, prev_W.data(), w_size*sizeof(ffm_float));
                        cout << endl << "Auto-stop. Use model at " << iter-1 << "th iteration." << endl;
                        break;
                    }
                    else
                    {
                        memcpy(prev_W.data(), model->W, w_size*sizeof(ffm_float));
                        best_auc_loss = va_loss; 
                    }
                }
            }
            cout << endl;
        }
    }


    fclose(f_tr);
    if(!va_path.empty())
        fclose(f_va);

#if defined USEOMP
    omp_set_num_threads(old_nr_threads);
#endif

    return model;
}

} // unnamed namespace

ffm_problem* ffm_read_problem(char const *path)
{
    if(strlen(path) == 0)
        return nullptr;

    FILE *f = fopen(path, "r");
    if(f == nullptr)
        return nullptr;

    ffm_problem *prob = new ffm_problem;
    prob->l = 0;
    prob->n = 0;
    prob->m = 0;
    prob->X = nullptr;
    prob->P = nullptr;
    prob->Y = nullptr;

    char line[kMaxLineSize];

    ffm_long nnz = 0;
    for(; fgets(line, kMaxLineSize, f) != nullptr; prob->l++)
    {
        strtok(line, " \t");
        for(; ; nnz++)
        {
            char *ptr = strtok(nullptr," \t");
            if(ptr == nullptr || *ptr == '\n')
                break;
        }
    }
    rewind(f);

    prob->X = new ffm_node[nnz];
    prob->P = new ffm_long[prob->l+1];
    prob->Y = new ffm_float[prob->l];

    ffm_long p = 0;
    prob->P[0] = 0;
    for(ffm_int i = 0; fgets(line, kMaxLineSize, f) != nullptr; i++)
    {
        char *y_char = strtok(line, " \t");
        ffm_float y = (atoi(y_char)>0)? 1.0f : -1.0f;
        prob->Y[i] = y;

        for(; ; p++)
        {
            char *field_char = strtok(nullptr,":");
            char *idx_char = strtok(nullptr,":");
            char *value_char = strtok(nullptr," \t");
            if(field_char == nullptr || *field_char == '\n')
                break;

            ffm_int field = atoi(field_char);
            ffm_int idx = atoi(idx_char);
            ffm_float value = atof(value_char);

            prob->m = max(prob->m, field+1);
            prob->n = max(prob->n, idx+1);
            
            prob->X[p].f = field;
            prob->X[p].j = idx;
            prob->X[p].v = value;
        }
        prob->P[i+1] = p;
    }

    fclose(f);

    return prob;
}

int ffm_read_problem_to_disk(char const *txt_path, char const *bin_path)
{
    FILE *f_txt = fopen(txt_path, "r");
    if(f_txt == nullptr)
        return 1;

    FILE *f_bin = fopen(bin_path, "wb");
    if(f_bin == nullptr)
        return 1;

    vector<char> line(kMaxLineSize);

    ffm_int m = 0;
    ffm_int n = 0;
    ffm_int max_l = 0;
    ffm_long max_nnz = 0;
    ffm_long p = 0;

    vector<ffm_float> Y;
    vector<ffm_float> R;
    vector<ffm_long> P(1, 0);
    vector<ffm_node> X;

    auto write_chunk = [&] ()
    {
        ffm_int l = Y.size();
        ffm_long nnz = P[l];

        max_l = max(max_l, l);
        max_nnz = max(max_nnz, nnz);

        fwrite(&l, sizeof(ffm_int), 1, f_bin);
        fwrite(Y.data(), sizeof(ffm_float), l, f_bin);
        fwrite(R.data(), sizeof(ffm_float), l, f_bin);
        fwrite(P.data(), sizeof(ffm_long), l+1, f_bin);
        fwrite(X.data(), sizeof(ffm_node), nnz, f_bin);

        Y.clear();
        R.clear();
        P.assign(1, 0);
        X.clear();
        p = 0;
    };

    fwrite(&m, sizeof(ffm_int), 1, f_bin);
    fwrite(&n, sizeof(ffm_int), 1, f_bin);
    fwrite(&max_l, sizeof(ffm_int), 1, f_bin);
    fwrite(&max_nnz, sizeof(ffm_long), 1, f_bin);

    while(fgets(line.data(), kMaxLineSize, f_txt))
    {
        char *y_char = strtok(line.data(), " \t");

        ffm_float y = (atoi(y_char)>0)? 1.0f : -1.0f;

        ffm_float scale = 0;
        for(; ; p++)
        {
            char *field_char = strtok(nullptr,":");
            char *idx_char = strtok(nullptr,":");
            char *value_char = strtok(nullptr," \t");
            if(field_char == nullptr || *field_char == '\n')
                break;

            ffm_node N;
            N.f = atoi(field_char);
            N.j = atoi(idx_char);
            N.v = atof(value_char);

            X.push_back(N);

            m = max(m, N.f+1);
            n = max(n, N.j+1);

            scale += N.v*N.v;
        }
        scale = 1/scale;

        Y.push_back(y);
        R.push_back(scale);
        P.push_back(p);

        if(X.size() > (size_t)kCHUNK_SIZE)
            write_chunk(); 
    }
    write_chunk(); 
    write_chunk(); 

    rewind(f_bin);
    fwrite(&m, sizeof(ffm_int), 1, f_bin);
    fwrite(&n, sizeof(ffm_int), 1, f_bin);
    fwrite(&max_l, sizeof(ffm_int), 1, f_bin);
    fwrite(&max_nnz, sizeof(ffm_long), 1, f_bin);

    fclose(f_bin);
    fclose(f_txt);

    return 0;
}

void ffm_destroy_problem(ffm_problem **prob)
{
    if(prob == nullptr || *prob == nullptr)
        return;
    delete[] (*prob)->X;
    delete[] (*prob)->P;
    delete[] (*prob)->Y;
    delete *prob;
    *prob = nullptr;
}

ffm_int ffm_save_model(ffm_model *model, char const *path, ffm_int &param_k)
{
    ofstream f_out(path);
    if(!f_out.is_open())
        return 1;

    f_out << "n " << model->n << "\n";
    f_out << "m " << model->m << "\n";
    f_out << "k " << model->k << "\n";
    f_out << "b "  << model->tr_avg << "\n";
    f_out << "normalization " << model->normalization << "\n";
    f_out << "nr_tr_feats " << model->tr_feat_idx.size() << "\n";
    f_out << "tr_feats" << " ";
    for (auto tr_f : model->tr_feat_idx)
        f_out << tr_f << " ";
    f_out << "\n";

    ffm_float *ptr = model->W;
    for(ffm_int j = 0; j < model->n; j++)
    {
        for(ffm_int f = 0; f < model->m; f++)
        {
            f_out << "w" << j << "," << f << " ";
            for(ffm_int d = 0; d < param_k; d++, ptr++)
                f_out << *ptr << " ";
            f_out << "\n";
            for(ffm_int d = param_k; d < model->k; d++, ptr++){ }
            f_out << "n" << j << "," << f << " ";
            for(ffm_int d = model->k; d < model->k+param_k; d++, ptr++)
                f_out << *ptr << " ";
            f_out << "\n";
        }
    }

    for(ffm_int j = 0; j < kBIN_SIZE; j++) {
        f_out << "wp" << j << " ";
        f_out << model->WP[j*2] << "\n";
        f_out << "np" << j << " ";
        f_out << model->WP[j*2+1] << "\n";
    }

    for(ffm_int j = 0; j < model->n; j++) {
        f_out << "wl" << j << " ";
        f_out << model->WL[j*2] << "\n";
        f_out << "nl" << j << " ";
        f_out << model->WL[j*2+1] << "\n";
    }

    f_out << "wb " << model->WB[0] << "\n";
    f_out << "nb " << model->WB[1] << "\n";

    return 0;
}

ffm_model* ffm_load_model(char const *path)
{
    ifstream f_in(path);
    if(!f_in.is_open())
        return nullptr;

    string dummy;

    ffm_model *model = new ffm_model;
    model->W = nullptr;

    ffm_int k;

    f_in >> dummy >> model->n >> dummy >> model->m >> dummy >> k 
         >> dummy >> model->tr_avg >> dummy >> model->normalization;

    ffm_int k_aligned = (ffm_int)ceil((ffm_double)k/kALIGN)*kALIGN;
    model->k = k_aligned;

    ffm_int nr_tr_feats;
    f_in >> dummy >> nr_tr_feats;
    f_in >> dummy;
    ffm_int feat_idx;
    for (ffm_int i = 0; i < nr_tr_feats; i++)
    {
        f_in >> feat_idx;
        model->tr_feat_idx.insert(feat_idx);
    }
    try
    {
        model->W = malloc_aligned_float((ffm_long)model->m*model->n*k_aligned*2);
        model->WL = malloc_aligned_float(model->n*2);
        model->WB = malloc_aligned_float(2);
        model->WP = malloc_aligned_float(2*kBIN_SIZE);
    }
    catch(bad_alloc const &e)
    {
        ffm_destroy_model(&model);
        return nullptr;
    }

    ffm_float *ptr = model->W;
    for(ffm_int j = 0; j < model->n; j++)
    {
        for(ffm_int f = 0; f < model->m; f++)
        {
            f_in >> dummy;
            for(ffm_int d = 0; d < k; d++, ptr++)
                f_in >> *ptr;

            for(ffm_int d = k; d < k_aligned; d++, ptr++)
                *ptr = 0;

            f_in >> dummy;
            for(ffm_int d = k_aligned; d < 2*k_aligned; d++, ptr++)
                f_in >> *ptr;
        }
    }

    ptr = model->WP;
    for(ffm_int j = 0; j < kBIN_SIZE; j++, ptr++) {
        f_in >> dummy;
        f_in >> *ptr;
        ptr++;
        f_in >> dummy;
        f_in >> *ptr;
    }

    ptr = model->WL;
    for(ffm_int j = 0; j < model->n; j++, ptr++) {
        f_in >> dummy;
        f_in >> *ptr;
        ptr++;
        f_in >> dummy;
        f_in >> *ptr;
    }

    ptr = model->WB;
    f_in >> dummy;
    f_in >> *ptr;
    ptr++;
    f_in >> dummy;
    f_in >> *ptr;
    return model;
}

void ffm_destroy_model(ffm_model **model)
{
    if(model == nullptr || *model == nullptr)
        return;
#ifdef _WIN32
    _aligned_free((*model)->W);
    _aligned_free((Pmodel)->WL);
    _aligned_free((*model)->WB);
    _aligned_free((*model)->WP);
#else
    free((*model)->W);
    free((*model)->WL);
    free((*model)->WB);
    free((*model)->WP);
#endif
    delete *model;
    *model = nullptr;
}

ffm_parameter ffm_get_default_param()
{
    ffm_parameter param;

    param.eta = 0.2;
    param.lambda = 0.00002;
    param.nr_iters = 15;
    param.k = 4;
    param.nr_threads = 1;
    param.quiet = false;
    param.normalization = true;
    param.random = true;
    param.auto_stop = false;

    return param;
}

ffm_model* ffm_train_with_validation(ffm_problem *tr, ffm_problem *va, ffm_parameter param, ffm_model *raw_model)
{
    vector<ffm_int> order(tr->l);
    for(ffm_int i = 0; i < tr->l; i++)
        order[i] = i;

    shared_ptr<ffm_model> model = train(tr, order, param, raw_model, va);

    ffm_model *model_ret = new ffm_model;

    model_ret->n = model->n;
    model_ret->m = model->m;
    model_ret->k = model->k;
    model_ret->normalization = model->normalization;
    model_ret->tr_avg = model->tr_avg;
    model_ret->tr_feat_idx = model->tr_feat_idx;

    model_ret->W = model->W;
    model_ret->WL = model->WL;
    model_ret->WB = model->WB;
    model_ret->WP = model->WP;

    model->W = nullptr;
    model->WL = nullptr;
    model->WB = nullptr;
    model->WP = nullptr;

    return model_ret;
}

ffm_model* ffm_train(ffm_problem *prob, ffm_parameter param, ffm_model *model)
{
    return ffm_train_with_validation(prob, nullptr, param, model);
}

ffm_model* ffm_train_with_validation_on_disk(
    char const *tr_path,
    char const *va_path,
    ffm_parameter param,
    ffm_model *raw_model)
{
    shared_ptr<ffm_model> model = train_on_disk(tr_path, va_path, param, raw_model);

    ffm_model *model_ret = new ffm_model;

    model_ret->n = model->n;
    model_ret->m = model->m;
    model_ret->k = model->k;
    model_ret->normalization = model->normalization;
    model_ret->tr_avg = model->tr_avg;
    model_ret->tr_feat_idx = model->tr_feat_idx;

    model_ret->W = model->W;
    model_ret->WL = model->WL;
    model_ret->WB = model->WB;
    model_ret->WP = model->WP;

    model->W = nullptr;
    model->WL = nullptr;
    model->WB = nullptr;
    model->WP = nullptr;

    return model_ret;
}

ffm_model* ffm_train_on_disk(char const *prob_path, ffm_parameter param, ffm_model *raw_model)
{
    return ffm_train_with_validation_on_disk(prob_path, "", param, raw_model);
}

ffm_float ffm_predict(ffm_node *begin, ffm_node *end, ffm_model *model)
{
    ffm_float r = 1;
    if(model->normalization)
    {
        r = 0;
        for(ffm_node *N = begin; N != end; N++)
            r += N->v*N->v; 
        r = 1/r;
    }

    ffm_int nr_nnz = 0;
    ffm_int nr_missed = 0;
    for (ffm_node *N = begin; N != end; N++)
    {
        nr_nnz++;
        ffm_int j = N->j;
        if (!model->tr_feat_idx.count(j))
            nr_missed++;
    }

    ffm_float missed_bias = 0.0;

    if (!nr_missed)
        missed_bias = 0.0;
    else
        missed_bias = model->tr_avg*(nr_missed*(nr_missed-1))/(nr_nnz*(nr_nnz-1));


    ffm_float t = wTx(begin, end, r, *model);
    t += missed_bias;
    return 1/(1+exp(-t));
}

ffm_float ffm_cross_validation(
    ffm_problem *prob, 
    ffm_int nr_folds,
    ffm_parameter param,
    ffm_model *raw_model)
{
#if defined USEOMP
    ffm_int old_nr_threads = omp_get_num_threads();
    omp_set_num_threads(param.nr_threads);
#endif

    bool quiet = param.quiet;
    param.quiet = true;

    vector<ffm_int> order(prob->l);
    for(ffm_int i = 0; i < prob->l; i++)
        order[i] = i;
    random_shuffle(order.begin(), order.end());

    if(!quiet)
    {
        cout.width(4);
        cout << "fold";
        cout.width(13);
        cout << "logloss";
        cout << endl;
    }

    ffm_double loss = 0;
    ffm_int nr_instance_per_fold = prob->l/nr_folds;
    for(ffm_int fold = 0; fold < nr_folds; fold++)
    {
        ffm_int begin = fold*nr_instance_per_fold;
        ffm_int end = min(begin + nr_instance_per_fold, prob->l);

        vector<ffm_int> order1;
        for(ffm_int i = 0; i < begin; i++)
            order1.push_back(order[i]);
        for(ffm_int i = end; i < prob->l; i++)
            order1.push_back(order[i]);

        shared_ptr<ffm_model> model = train(prob, order1, param, raw_model);

        ffm_double loss1 = 0;
#if defined USEOMP
#pragma omp parallel for schedule(static) reduction(+: loss1)
#endif
        for(ffm_int ii = begin; ii < end; ii++)
        {
            ffm_int i = order[ii];

            ffm_float y = prob->Y[i];
            
            ffm_node *begin = &prob->X[prob->P[i]];

            ffm_node *end = &prob->X[prob->P[i+1]];

            ffm_float y_bar = ffm_predict(begin, end, model.get());

            loss1 -= y==1? log(y_bar) : log(1-y_bar);
        }
        loss += loss1;

        if(!quiet)
        {
            cout.width(4);
            cout << fold;
            cout.width(13);
            cout << fixed << setprecision(4) << loss1 / (end-begin);
            cout << endl;
        }
    }

    if(!quiet)
    {
        cout.width(17);
        cout.fill('=');
        cout << "" << endl;
        cout.fill(' ');
        cout.width(4);
        cout << "avg";
        cout.width(13);
        cout << fixed << setprecision(4) << loss/prob->l;
        cout << endl;
    }

#if defined USEOMP
    omp_set_num_threads(old_nr_threads);
#endif

    return loss/prob->l;
}

} // namespace ffm
