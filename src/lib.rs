pub mod exactsplitter;
mod node;
pub mod objective;
mod splitter;
pub mod tree;

pub mod data;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
