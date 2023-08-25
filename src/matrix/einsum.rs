use crate::{BasicMatrix, MatrixFull, matrix_blas_lapack::_dgemm_full};

/// Supported einsum: 
/// * "ij,j->ij" 
/// * "ip,ip->p" 
/// * "i,j->ij"  
/// * "ij,jk->ik"
/// * "ij,ij->j" 
/// * "i,ij->ij"
/// * "ij,ij->i"
pub fn _einsum_general<'a, A, B>(mat_a: &A, mat_b: &B, opt: &str)  -> MatrixFull<f64> 
    where A: BasicMatrix<'a,f64>,
          B: BasicMatrix<'a,f64>
    {
    match opt {
        "ij,j->ij" => _einsum_01_general(mat_a, mat_b),
        "ip,ip->p" => _einsum_02_general(mat_a, mat_b),
        "i,j->ij"  => _einsum_03_general(mat_a, mat_b),
        "ij,jk->ik" => _einsum_04_general(mat_a, mat_b),
        "ij,ij->j" => _einsum_05_general(mat_a, mat_b),
        "i,ij->ij" => _einsum_06_general(mat_a, mat_b),
        "ij,ij->i" => _einsum_07_general(mat_a, mat_b),
        _  => panic!("Not implemented for einsum: {}", opt),
    }
}

#[inline]
/// einsum ip, ip -> p
pub fn _einsum_02_general<'a, A, B>(mat_a: &A, mat_b: &B) -> MatrixFull<f64>
    where A: BasicMatrix<'a,f64>,
          B: BasicMatrix<'a,f64>
    {
    let a_y = mat_a.size()[1];
    let b_y = mat_b.size()[1];
    let a_x = mat_a.size()[0];
    let b_x = mat_b.size()[0];
    if (a_x == 0 || b_x ==0) {return MatrixFull::new([a_y.min(b_y),1],0.0)};
    let mut out_vec = vec![0.0;a_y.min(b_y)];

    mat_a.data_ref().unwrap().chunks_exact(a_x).zip(mat_b.data_ref().unwrap().chunks_exact(b_x))   
    .zip(out_vec.iter_mut())
    .for_each(|((mat_a_p,mat_b_p),out_vec_p)| {
        *out_vec_p = mat_a_p.iter().zip(mat_b_p.iter())
            .fold(0.0, |acc, (mat_a_ip, mat_b_ip)| 
            {acc + mat_a_ip*mat_b_ip});
    });
    MatrixFull::from_vec([a_y.min(b_y),1],out_vec).unwrap()
}
#[inline]
/// einsum: ij,j->ij
pub fn _einsum_01_general<'a, A, B>(mat_a: &A, mat_b: &B) -> MatrixFull<f64>
    where A: BasicMatrix<'a,f64>,
          B: BasicMatrix<'a,f64>
    {
    let vec_b = mat_b.data_ref().unwrap();
    let i_len = mat_a.size()[0];
    let j_len = vec_b.len();
    if (i_len == 0 || j_len ==0) {return MatrixFull::new([i_len,j_len],0.0)};
    let mut om = MatrixFull::new([i_len,j_len],0.0);

    om.iter_columns_full_mut().zip(mat_a.data_ref().unwrap().chunks_exact(i_len))
    .map(|(om_j,mat_a_j)| {(om_j,mat_a_j)})
    .zip(vec_b.iter())
    .for_each(|((om_j,mat_a_j),vec_b_j)| {
        om_j.iter_mut().zip(mat_a_j.iter()).for_each(|(om_ij,mat_a_ij)| {
            *om_ij = *mat_a_ij*vec_b_j
        });
    });
    om 
}

/// i, j -> ij <br>
/// vec_a: column vec of i rows, vec_b: row vec of j columns <br>
/// produces matrix of i,j 
pub fn _einsum_03_general<'a, A, B>(mat_a: &A, mat_b: &B) -> MatrixFull<f64>
    where A: BasicMatrix<'a,f64>,
          B: BasicMatrix<'a,f64>
    {

    let i_len = mat_a.data_ref().unwrap().len();
    let j_len = mat_b.data_ref().unwrap().len();
    if (i_len == 0 || j_len ==0) {return MatrixFull::new([i_len,j_len],0.0)};
    let mut om = MatrixFull::new([i_len,j_len],0.0);

    om.iter_columns_full_mut().zip(mat_b.data_ref().unwrap().iter())
    .map(|(om_j, vec_b)| {(om_j, vec_b)})
    .for_each(|(om_j, vec_b)| {
        om_j.iter_mut().zip(mat_a.data_ref().unwrap().iter()).for_each(|(om_ij, vec_a)| {
            *om_ij = *vec_a*vec_b
        });
    });

    om
}

/// ij,jk->ik <br>
/// simply call a dgemm function
pub fn _einsum_04_general<'a, A, B>(mat_a: &A, mat_b: &B) -> MatrixFull<f64>
    where A: BasicMatrix<'a,f64>,
          B: BasicMatrix<'a,f64>
    {
        let mut mat_c = MatrixFull::new([mat_a.size()[0],mat_b.size()[1]],0.0);
        _dgemm_full(mat_a,'N',mat_b,'N',&mut mat_c,1.0,0.0);
        mat_c
}

/// ij,ij->j <br>
/// simply perform ij,ij->jj and get diagonal terms of jj
pub fn _einsum_05_general<'a, A, B> (mat_a: &A, mat_b: &B) -> MatrixFull<f64>
    where A: BasicMatrix<'a,f64>,
          B: BasicMatrix<'a,f64>
    {
        let mut mat_c = MatrixFull::new([mat_a.size()[1];2],0.0);

        if (mat_a.size()[0] == mat_b.size()[0] && mat_a.size()[1] == mat_b.size()[1]) {
            // ij,ij
            _dgemm_full(mat_a,'T',mat_b,'N',&mut mat_c,1.0,0.0);
        } else if (mat_a.size()[0] == mat_b.size()[1] && mat_a.size()[1] == mat_b.size()[0])  {
            // ij,ji
            _dgemm_full(mat_b,'N',mat_a,'N',&mut mat_c,1.0,0.0);
        } else {
            panic!("Incompitable matrix size of mat_a and mat_b for einsum ij,ij->j")
        }
        let diag = mat_c.get_diagonal_terms().unwrap();
        let result: Vec<f64> = diag.into_iter().map(|v| v.clone()).collect();
        MatrixFull::from_vec([result.len(),1],result).unwrap()
        
}

/// i,ij->ij <br>
/// Multiply every i of ij by i  <br>
/// |1 2 3|  |1 2 3|    =>   |1 4  9 | 
///          |4 5 6|         |4 10 18|
/// size of mat_a must be [i,1] (column vector)
pub fn _einsum_06_general<'a, A, B> (mat_a: &A, mat_b: &B) -> MatrixFull<f64>
    where A: BasicMatrix<'a,f64>,
          B: BasicMatrix<'a,f64>
    {
        let i = mat_b.size()[0];
        let j = mat_b.size()[1];
        //let mut mat_c = MatrixFull::new([i,j],0.0);
        let mut data = vec![];
        if (mat_a.size()[1] != 1) {
            panic!("Error: mat_a is not a column vector");
        }

        if (mat_a.size()[0] == mat_b.size()[0]) {
            // i,ij
            data = mat_b.data_ref().unwrap().chunks_exact(i)
                    .map(|bchunk| bchunk.into_iter()
                        .zip(mat_a.data_ref().unwrap()).map(|(b,a)| b*a).collect::<Vec<f64>>()).flatten().collect::<Vec<f64>>();

        } else if (mat_a.size()[0] == mat_b.size()[1])  {
            // j,ij
            data = mat_b.data_ref().unwrap().chunks_exact(i).zip(mat_a.data_ref().unwrap())
                    .map(|(bchunk,a)| bchunk.into_iter().map(|b| b*a).collect::<Vec<f64>>()).flatten().collect::<Vec<f64>>();
        } else {
            panic!("Incompitable matrix size of mat_a and mat_b for einsum i,ij->ij")
        }

        let mat_c = MatrixFull::from_vec([i,j], data).unwrap();
        mat_c
}


/// ij,ij->i <br>
/// simply perform ij,ij->ii and get diagonal terms of ii
pub fn _einsum_07_general<'a, A, B> (mat_a: &A, mat_b: &B) -> MatrixFull<f64>
    where A: BasicMatrix<'a,f64>,
          B: BasicMatrix<'a,f64>
    {
        let mut mat_c = MatrixFull::new([mat_a.size()[1];2],0.0);

        if (mat_a.size()[0] == mat_b.size()[0] && mat_a.size()[1] == mat_b.size()[1]) {
            // ij,ij
            _dgemm_full(mat_a,'N',mat_b,'T',&mut mat_c,1.0,0.0);
        } else if (mat_a.size()[0] == mat_b.size()[1] && mat_a.size()[1] == mat_b.size()[0])  {
            // ij,ji
            _dgemm_full(mat_a,'N',mat_b,'N',&mut mat_c,1.0,0.0);
        } else {
            panic!("Incompitable matrix size of mat_a and mat_b for einsum ij,ij->j")
        }
        let diag = mat_c.get_diagonal_terms().unwrap();
        let result: Vec<f64> = diag.into_iter().map(|v| v.clone()).collect();
        MatrixFull::from_vec([result.len(),1],result).unwrap()
        
}


#[test]
fn test_einsum06() {
    let data_b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let data_a = vec![1.0, 2.0];
    let mat_a = MatrixFull::from_vec([2,1], data_a).unwrap();
    let mat_b = MatrixFull::from_vec([3,2], data_b).unwrap();
    let mat_c = _einsum_06_general(&mat_a, &mat_b);
    mat_c.formated_output(10, "full");
}