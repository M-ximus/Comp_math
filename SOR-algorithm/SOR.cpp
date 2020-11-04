#include <cstdio>
#include <cstdlib>
#include <errno.h>
#include <cmath>

const double PRECISION = 0.0001;
const double OMEGA     = 1.5;

enum error{
    E_ERROR    = -1,
    E_BADARGS  = -2,
    E_BADALLOC = -3,
    E_FPU      = -4
};

struct sq_matrix{
    int dim_;
    double* arr_data_;
    double** data_;
};

int constr_matrix(sq_matrix* matr, int dim, FILE* in);
int destr_matrix(sq_matrix* matr);

struct vector{
    int dim_;
    double* data_;
};

int constr_vector(vector* vec, int dim);
int destr_vector(vector* vec);
int print_vector(vector* vec);

int mul_sq_matrix_to_vector(sq_matrix* matr, vector* vec, vector* res);
int SOR(sq_matrix* A, vector* zero, vector* f, double omega, double prec, vector* res);

int sqrt_s(double num, double* res);
static int calc_euclid_norm_of_residual(sq_matrix* A, vector* u, vector* f, double* res);


int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        printf("[main] Bad num of arguments\n");
        exit(EXIT_FAILURE);
    }

    errno = 0;
    FILE* in_file = fopen(argv[1], "r");
    if (in_file == nullptr)
    {
        perror("[main] Opening input file error\n");
        exit(EXIT_FAILURE);
    }

    int dim = 0;
    int ret = fscanf(in_file, "%d\n", &dim);
    if (ret != 1)
    {
        perror("[main] Scanning dimention of matrix error\n");
        exit(EXIT_FAILURE);
    }

    sq_matrix A_matrix;
    ret = constr_matrix(&A_matrix, dim, in_file);
    if (ret < 0)
    {
        printf("[main] constructig matrix error\n");
        exit(EXIT_FAILURE);
    }

    vector root;
    ret = constr_vector(&root, dim);
    if (ret < 0)
    {
        printf("[main] constructing root error\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < dim; i++)
        root.data_[i] = static_cast<double>(i + 1);

    vector f_vec;
    ret = constr_vector(&f_vec, dim);
    if (ret < 0)
    {
        printf("[main] constructing f_vec error\n");
        exit(EXIT_FAILURE);
    }


    ret = mul_sq_matrix_to_vector(&A_matrix, &root, &f_vec);
    if (ret < 0)
    {
        printf("[main] multiplication A*x = f return error\n");
        exit(EXIT_FAILURE);
    }

    vector iter_root;
    ret = constr_vector(&iter_root, dim);
    if (ret < 0)
    {
        printf("[main] constructing iterational root error\n");
        exit(EXIT_FAILURE);
    }

    vector zero;
    ret = constr_vector(&zero, dim);
    if (ret < 0)
    {
        printf("[main] constructing zero value error\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < dim; i++)
        zero.data_[i] = 0;

    ret = SOR(&A_matrix, &zero, &f_vec, OMEGA, PRECISION, &iter_root);
    if (ret < 0)
    {
        printf("[main] SOR method return error\n");
        exit(EXIT_FAILURE);
    }

    ret = print_vector(&iter_root);
    if (ret < 0)
    {
        printf("[main] print iter root error\n");
        exit(EXIT_FAILURE);
    }

    destr_matrix(&A_matrix);
    destr_vector(&root);
    destr_vector(&f_vec);
    destr_vector(&iter_root);

    return 0;
}

int sqrt_s(double num, double* res)
{
    if (res == nullptr)
    {
        printf("[sqrt_s] out res == nullptr\n");
        return E_BADARGS;
    }

    if (num < 0)
    {
        printf("[sqrt_s] num < 0\n");
        return E_FPU;
    }

    if (num == 0.0)
    {
        *res = 0.0;
        return 0;
    }

    double x0 = num;
    double prev = x0;
    do {
        prev = x0;
        if (x0 <= 0.0)
        {
            printf("[sqrt_s] computational error\n");
            return E_FPU;
        }

        x0 = 0.5 * (x0 + num / x0);
    } while(prev != x0);

    *res = x0;

    return 0;
}

static int calc_euclid_norm_of_residual(sq_matrix* A, vector* u, vector* f, double* res)
{
    if (A == nullptr || u == nullptr || f == nullptr || A->data_ == nullptr||
        res == nullptr || A->arr_data_ == nullptr || u->data_ == nullptr ||
        f->data_ == nullptr || A->dim_ != u->dim_ || A->dim_ != f->dim_)
    {
        printf("[calc_euclid_norm_of_residual] bad args\n");
        return E_BADARGS;
    }

    vector temp;
    int ret = constr_vector(&temp, A->dim_);
    if (ret < 0)
    {
        printf("[calc_euclid_norm_of_residual] construct temple vector error\n");
        return E_ERROR;
    }

    ret = mul_sq_matrix_to_vector(A, u, &temp);
    if (ret < 0)
    {
        printf("[calc_euclid_norm_of_residual] multiplication A*u error\n");
        return E_ERROR;
    }

    for (int i = 0; i < temp.dim_; i++)
        temp.data_[i] = f->data_[i] - temp.data_[i];

    double sum = 0.0;
    for (int i = 0; i < temp.dim_; i++)
        sum += temp.data_[i] * temp.data_[i];

    double result = 0.0;
    ret = sqrt_s(sum, &result);
    if (ret < 0)
    {
        printf("[calc_euclid_norm_of_residual] sqrt of sum\n");
        return E_ERROR;
    }

    *res = result;

    destr_vector(&temp);

    return 0;
}

int SOR(sq_matrix* A, vector* zero, vector* f, double omega, double prec, vector* res)
{
    if (A == nullptr || A->data_ == nullptr || A->arr_data_ == nullptr)
    {
        printf("[SOR] input matrix A == nullptr\n");
        return E_BADARGS;
    }

    if (zero == nullptr || zero->data_ == nullptr)
    {
        printf("[SOR] input zero vector == nullptr\n");
        return E_BADARGS;
    }

    if (f == nullptr || f->data_ == nullptr)
    {
        printf("[SOR] input f vector == nullptr\n");
        return E_BADARGS;
    }

    if (res == nullptr || res->data_ == nullptr)
    {
        printf("[SOR] out res vector == nullptr\n");
        return E_BADARGS;
    }

    if (A->dim_ != f->dim_ || A->dim_ != res->dim_ || A->dim_ != zero->dim_)
    {
        printf("[SOR] different dimentios\n");
        return E_BADARGS;
    }

    if (!std::isfinite(omega) || !std::isfinite(prec))
    {
        printf("[SOR] input params are nan\n");
        return E_BADARGS;
    }

    vector temp;
    int ret = constr_vector(&temp, A->dim_);
    if (ret < 0)
    {
        printf("[SOR] constructing temp vector error\n");
        return E_ERROR;
    }

    for (int i = 0; i < temp.dim_; i++)
        temp.data_[i] = zero->data_[i];

    double norm = 0.0;
    int num_of_iteration = 1;
    do{
        for (int row = 0; row < temp.dim_; row++)
        {
            temp.data_[row] = (1 - omega) * temp.data_[row];

            if (A->data_[row][row] == 0.0)
            {
                printf("[SOR] diag element (%d %d) == 0\n", row, row);
                destr_vector(&temp);
                return E_ERROR;
            }

            double sum = f->data_[row];

            for (int col = 0; col < row; col++)
                sum -= A->data_[row][col] * temp.data_[col]; // changed

            for (int col = row + 1; col < temp.dim_; col++)
                sum -= A->data_[row][col] * temp.data_[col]; // not changed

            sum *= omega / A->data_[row][row];
            temp.data_[row] += sum;
        }

        ret = calc_euclid_norm_of_residual(A, &temp, f, &norm);
        if (ret < 0)
        {
            printf("[SOR] calc euclid norm error\n");
            destr_vector(&temp);
            return E_ERROR;
        }
        printf("%d euclidus norm of residual = %lg\n", num_of_iteration, norm);

        num_of_iteration++;
    } while (norm > prec);

    for (int i = 0; i < temp.dim_; i++)
        res->data_[i] = temp.data_[i];

    destr_vector(&temp);

    return 0;
}

int constr_matrix(sq_matrix* matr, int dim, FILE* input)
{
    if (matr == nullptr)
    {
        printf("[constr_matrix] input matrix == nullptr\n");
        return E_BADARGS;
    }

    if (input == nullptr)
    {
        printf("[constr_matrix] input file == nullptr\n");
        return E_BADARGS;
    }

    if (dim < 0)
    {
        printf("[constr_matrix] dim < 0\n");
        return E_BADARGS;
    }

    errno = 0;

    double* arr_data = (double*)calloc(dim * dim, sizeof(*arr_data));
    if (arr_data == nullptr)
    {
        perror("[constr_matrix] bad alloc of element array\n");
        return E_BADALLOC;
    }

    double** data = (double**)calloc(dim, sizeof(*data));
    if (data == nullptr)
    {
        perror("[constr_matrix] bad alloc of strings array\n");
        return E_BADALLOC;
    }

    for (int row = 0; row < dim; row++)
    {
        data[row] = arr_data + row * dim;
        for (int col = 0; col < dim; col++)
        {
            int ret = fscanf(input, "%lg", &(data[row][col]));
            if (ret < 0)
            {
                printf("[constr_matrix] scanning element in row = %d, col = %d error\n", row, col);
                free(arr_data);
                free(data);
                return E_ERROR;
            }
        }
    }

    matr->dim_      = dim;
    matr->arr_data_ = arr_data;
    matr->data_     = data;

    return 0;
}

int constr_vector(vector* vec, int dim)
{
    if (vec == nullptr)
    {
        printf("[constr_vector] input vec == nullptr\n");
        return E_BADARGS;
    }

    if (dim < 0)
    {
        printf("[constr_vector] input dim < 0\n");
        return E_BADARGS;
    }

    errno = 0;

    double* data = (double*)calloc(dim, sizeof(*data));
    if (data == nullptr)
    {
        perror("[constr_vector] alloc data buffer error\n");
        return E_BADALLOC;
    }

    vec->dim_  = dim;
    vec->data_ = data;

    return 0;
}

int mul_sq_matrix_to_vector(sq_matrix* matr, vector* vec, vector* res)
{
    if (matr == nullptr || matr->data_ == nullptr || matr->arr_data_ == nullptr)
    {
        printf("[mul_sq_matrix_to_vector] input matr == nullptr\n");
        return E_BADARGS;
    }

    if (vec == nullptr || vec->data_ == nullptr)
    {
        printf("[mul_sq_matrix_to_vector] input vec == nullptr\n");
        return E_BADARGS;
    }

    if (res == nullptr || res->data_ == nullptr)
    {
        printf("[mul_sq_matrix_to_vector] out res == nullptr\n");
        return E_BADARGS;
    }

    if (matr->dim_ != vec->dim_)
    {
        printf("[mul_sq_matrix_to_vector] Bad format of parametrs matrix dim %d != vector dim %d\n", matr->dim_, vec->dim_);
        return E_ERROR;
    }

    for (int row = 0; row < vec->dim_; row++)
    {
        for(int i = 0; i < vec->dim_; i++)
            res->data_[row] += matr->data_[row][i] * vec->data_[i];
    }

    return 0;
}

int destr_matrix(sq_matrix* matr)
{
    if (matr == nullptr)
    {
        printf("[destr_matrix] input matr == nullptr\n");
        return E_BADARGS;
    }

    free(matr->data_);
    free(matr->arr_data_);
    matr->dim_ = -1;

    return 0;
}

int destr_vector(vector* vec)
{
    if (vec == nullptr)
    {
        printf("[destr_vector] input vector == nullptr \n");
        return E_BADARGS;
    }

    free(vec->data_);
    vec->dim_ = 0;

    return 0;
}

int print_vector(vector* vec)
{
    if (vec == nullptr)
    {
        printf("[print_vector] input vec == nullptr\n");
        return E_BADARGS;
    }

    for (int i = 0; i < vec->dim_; i++)
        printf("%lg\n", vec->data_[i]);

    return 0;
}
