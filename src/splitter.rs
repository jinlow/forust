use crate::data::{Matrix, MatrixData};
use crate::node::Node;

#[derive(Debug)]
pub struct SplitInfo<T> {
    pub split_gain: T,
    pub split_feature: usize,
    pub split_value: T,
    pub left_gain: T,
    pub left_cover: T,
    pub left_weight: T,
    pub left_idxs: Vec<usize>,
    pub right_gain: T,
    pub right_cover: T,
    pub right_weight: T,
    pub right_idxs: Vec<usize>,
}

pub trait Splitter<T>
where
    T: MatrixData<T>,
{
    fn get_l2(&self) -> T;

    fn get_learning_rate(&self) -> T;

    fn get_gamma(&self) -> T;

    fn best_split(
        &self,
        node: &mut Node<T>,
        data: &Matrix<T>,
        grad: &[T],
        hess: &[T],
    ) -> Option<SplitInfo<T>>;

    fn gain(&self, grad_sum: T, hess_sum: T) -> T {
        (grad_sum * grad_sum) / (hess_sum + self.get_l2())
    }

    fn weight(&self, grad_sum: T, hess_sum: T) -> T {
        -((grad_sum / (hess_sum + self.get_l2())) * self.get_learning_rate())
    }
}
