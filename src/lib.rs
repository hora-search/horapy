use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyList;
use real_hora::core::ann_index::ANNIndex;
use real_hora::core::ann_index::SerializableIndex;
use real_hora::core::metrics;
use real_hora::core::node;

pub enum KIdxType {
    String,
    Usize,
}

#[pyclass]
struct ANNNode {
    #[pyo3(get)]
    vectors: Vec<f32>, // the vectors;
    #[pyo3(get)]
    idx: usize, // data id, it can be any type;
}

fn transform_didx_type(src: &str) -> KIdxType {
    match src {
        "usize" => KIdxType::Usize,
        "string" => KIdxType::String,
        _ => KIdxType::Usize,
    }
}

// fn transform(src: &[(node::Node<f32, usize>, f32)]) -> Vec<(ANNNode, f32)> {
//     let dst: Vec<(ANNNode, f32)> = src
//         .iter()
//         .map(|i| {
//             (
//                 ANNNode {
//                     vectors: i.0.vectors().clone(),
//                     idx: *i.0.idx().as_ref().unwrap(),
//                 },
//                 i.1,
//             )
//         })
//         .collect();
//     dst
// }

fn metrics_transform(s: &str) -> metrics::Metric {
    match s {
        "angular" => metrics::Metric::Angular,
        "manhattan" => metrics::Metric::Manhattan,
        "dot_product" => metrics::Metric::DotProduct,
        "euclidean" => metrics::Metric::Euclidean,
        "cosine_similarity" => metrics::Metric::CosineSimilarity,
        _ => metrics::Metric::Unknown,
    }
}

#[macro_export]
macro_rules! inherit_ann_index_method {
    (  $ann_idx:ident,$type_expr: ty, $idx_type_expr: ty) => {
        #[pyclass]
        struct $ann_idx {
            _idx: Box<$type_expr>,
        }

        impl $ann_idx {
            fn add_node(&mut self, item: &node::Node<f32, $idx_type_expr>) -> PyResult<bool> {
                return match self._idx.add_node(item) {
                    Ok(()) => Ok(true),
                    Err(_e) => Ok(false), //TODO
                };
            }
            // fn node_search_k(
            //     &self,
            //     item: &node::Node<f32, $idx_type_expr>,
            //     k: usize,
            // ) -> PyResult<Vec<(ANNNode, f32)>> {
            //     Ok(transform(&self._idx.node_search_k(item, k))) //TODO: wrap argument
            // }

            fn node_search_k_ids(&self, item: &[f32], k: usize) -> PyResult<Vec<$idx_type_expr>> {
                Ok(self._idx.search(item, k))
            }
        }

        #[pymethods]
        impl $ann_idx {
            fn build(&mut self, py: Python, s: String) -> PyResult<bool> {
                py.allow_threads(|| self._idx.build(metrics_transform(&s)).unwrap());
                Ok(true)
            }
            fn add_without_idx(&mut self, pyvs: &PyList) -> PyResult<bool> {
                let mut vs = Vec::new();
                for i in pyvs.iter() {
                    vs.push(i.extract::<f32>().unwrap())
                }
                let n = node::Node::new(&vs);
                self.add_node(&n)
            }
            fn add(&mut self, pyvs: &PyList, idx: $idx_type_expr) -> PyResult<bool> {
                let mut vs = Vec::new();
                for i in pyvs.iter() {
                    vs.push(i.extract::<f32>().unwrap())
                }
                let n = node::Node::new_with_idx(&vs, idx);
                self.add_node(&n)
            }

            // fn search_k(&self, pyvs: &PyList, k: usize) -> PyResult<Vec<(ANNNode, f32)>> {
            //     let mut vs = Vec::new();
            //     for i in pyvs.iter() {
            //         vs.push(i.extract::<f32>().unwrap())
            //     }
            //     let n = node::Node::new(&vs);
            //     self.node_search_k(&n, k)
            // }

            fn search_np<'py>(
                &self,
                pyvs: PyReadonlyArray1<'py, f32>,
                k: usize,
            ) -> PyResult<Vec<$idx_type_expr>> {
                let vs: Vec<f32> = pyvs.as_array().into_iter().map(|i| i.clone()).collect();
                let res: Vec<$idx_type_expr> = self._idx.search(&vs, k);
                Ok(res)
            }

            fn search(&self, pyvs: &PyList, k: usize) -> PyResult<Vec<$idx_type_expr>> {
                let vs: Vec<f32> = pyvs.iter().map(|i| i.extract::<f32>().unwrap()).collect();
                self.node_search_k_ids(&vs, k)
            }

            fn name(&self) -> PyResult<String> {
                Ok(stringify!($ann_idx).to_string())
            }

            #[staticmethod]
            fn load(path: String) -> Self {
                $ann_idx {
                    _idx: Box::new(<$type_expr>::load(&path).unwrap()),
                }
            }

            fn dump(&mut self, path: String) {
                self._idx.dump(&path).unwrap();
            }
        }
    };
}

// inherit_ann_index_method!(BPTIndex, real_hora::index::rpt_idx::BPTIndex<f32, usize>);
// inherit_ann_index_method!(BruteForceIndex, real_hora::index::bruteforce_idx::BruteForceIndex<f32,usize>, usize);
// inherit_ann_index_method!(HNSWIndex, real_hora::index::hnsw_idx::HNSWIndex<f32, usize>,usize);
// inherit_ann_index_method!(PQIndex, real_hora::index::pq_idx::PQIndex<f32, usize>,usize);
// inherit_ann_index_method!(IVFPQIndex, real_hora::index::pq_idx::IVFPQIndex<f32, usize>,usize);
// inherit_ann_index_method!(SSGIndex, real_hora::index::ssg_idx::SSGIndex<f32, usize>,usize);

#[macro_export]
macro_rules! define_bruteforce_ann_index {
    (  $idx_name:ident, $idx_type_expr: ty) => {
        #[pymethods]
        impl $idx_name {
            #[new]
            fn new(dimension: usize) -> Self {
                $idx_name {
                    _idx: Box::new(real_hora::index::bruteforce_idx::BruteForceIndex::<
                        f32,
                        $idx_type_expr,
                    >::new(
                        dimension,
                        &real_hora::index::bruteforce_params::BruteForceParams::default(),
                    )),
                }
            }
        }
    };
}
inherit_ann_index_method!(BruteForceIndexUsize, real_hora::index::bruteforce_idx::BruteForceIndex<f32,usize>, usize);
define_bruteforce_ann_index!(BruteForceIndexUsize, usize);
inherit_ann_index_method!(BruteForceIndexStr, real_hora::index::bruteforce_idx::BruteForceIndex<f32,String>, String);
define_bruteforce_ann_index!(BruteForceIndexStr, String);

// #[pymethods]
// impl BPTIndex {
//     #[new]
//     fn new(dimension: usize, tree_num: i32, candidate_size: i32) -> Self {
//         BPTIndex {
//             _idx: Box::new(real_hora::index::bpt_idx::BPTIndex::<f32, usize>::new(
//                 dimension,
//                 real_hora::index::bpt_params::BPTParams::default()
//                     .tree_num(tree_num)
//                     .candidate_size(candidate_size),
//             )),
//         }
//     }
// }

#[macro_export]
macro_rules! define_hnsw_ann_index {
    (  $idx_name:ident, $idx_type_expr: ty) => {
        #[pymethods]
        impl $idx_name {
            #[new]
            fn new(
                dimension: usize,
                max_item: usize,
                n_neigh: usize,
                n_neigh0: usize,
                ef_build: usize,
                ef_search: usize,
                has_deletion: bool,
            ) -> Self {
                $idx_name {
                    _idx: Box::new(
                        real_hora::index::hnsw_idx::HNSWIndex::<f32, $idx_type_expr>::new(
                            dimension,
                            &real_hora::index::hnsw_params::HNSWParams::<f32>::default()
                                .max_item(max_item)
                                .n_neighbor(n_neigh)
                                .n_neighbor0(n_neigh0)
                                .ef_build(ef_build)
                                .ef_search(ef_search)
                                .has_deletion(has_deletion),
                        ),
                    ),
                }
            }
        }
    };
}

inherit_ann_index_method!(HNSWIndexUsize, real_hora::index::hnsw_idx::HNSWIndex<f32, usize>,usize);
define_hnsw_ann_index!(HNSWIndexUsize, usize);
inherit_ann_index_method!(HNSWIndexStr, real_hora::index::hnsw_idx::HNSWIndex<f32, String>,String);
define_hnsw_ann_index!(HNSWIndexStr, String);

#[macro_export]
macro_rules! define_pq_ann_index {
    (  $idx_name:ident, $idx_type_expr: ty) => {
        #[pymethods]
        impl $idx_name {
            #[new]
            fn new(dimension: usize, n_sub: usize, sub_bits: usize, train_epoch: usize) -> Self {
                $idx_name {
                    _idx: Box::new(
                        real_hora::index::pq_idx::PQIndex::<f32, $idx_type_expr>::new(
                            dimension,
                            &real_hora::index::pq_params::PQParams::default()
                                .n_sub(n_sub)
                                .sub_bits(sub_bits)
                                .train_epoch(train_epoch),
                        ),
                    ),
                }
            }
        }
    };
}
inherit_ann_index_method!(PQIndexUsize, real_hora::index::pq_idx::PQIndex<f32, usize>,usize);
define_pq_ann_index!(PQIndexUsize, usize);
inherit_ann_index_method!(PQIndexStr, real_hora::index::pq_idx::PQIndex<f32, String>,String);
define_pq_ann_index!(PQIndexStr, String);

#[macro_export]
macro_rules! define_ivfpq_ann_index {
    (  $idx_name:ident, $idx_type_expr: ty) => {
        #[pymethods]
        impl $idx_name {
            #[new]
            fn new(
                dimension: usize,
                n_sub: usize,
                sub_bits: usize,
                n_kmeans_center: usize,
                search_n_center: usize,
                train_epoch: usize,
            ) -> Self {
                $idx_name {
                    _idx: Box::new(
                        real_hora::index::pq_idx::IVFPQIndex::<f32, $idx_type_expr>::new(
                            dimension,
                            &real_hora::index::pq_params::IVFPQParams::default()
                                .n_sub(n_sub)
                                .sub_bits(sub_bits)
                                .n_kmeans_center(n_kmeans_center)
                                .search_n_center(search_n_center)
                                .train_epoch(train_epoch),
                        ),
                    ),
                }
            }
        }
    };
}

inherit_ann_index_method!(IVFPQIndexUsize, real_hora::index::pq_idx::IVFPQIndex<f32, usize>,usize);
define_ivfpq_ann_index!(IVFPQIndexUsize, usize);
inherit_ann_index_method!(IVFPQIndexStr, real_hora::index::pq_idx::IVFPQIndex<f32, String>,String);
define_ivfpq_ann_index!(IVFPQIndexStr, String);

#[macro_export]
macro_rules! define_ssg_ann_index {
    (  $idx_name:ident, $idx_type_expr: ty) => {
        #[pymethods]
        impl $idx_name {
            #[new]
            #[args(
                neighbor_neighbor_size = "100",
                init_k = "100",
                index_size = "100",
                angle = "30.0",
                root_size = "30"
            )]
            fn new(
                dimension: usize,
                neighbor_neighbor_size: usize,
                init_k: usize,
                index_size: usize,
                angle: f32,
                root_size: usize,
            ) -> Self {
                $idx_name {
                    _idx: Box::new(
                        real_hora::index::ssg_idx::SSGIndex::<f32, $idx_type_expr>::new(
                            dimension,
                            &real_hora::index::ssg_params::SSGParams::default()
                                .neighbor_neighbor_size(neighbor_neighbor_size)
                                .init_k(init_k)
                                .index_size(index_size)
                                .angle(angle)
                                .root_size(root_size),
                        ),
                    ),
                }
            }
        }
    };
}

inherit_ann_index_method!(SSGIndexUsize, real_hora::index::ssg_idx::SSGIndex<f32, usize>,usize);
define_ssg_ann_index!(SSGIndexUsize, usize);
inherit_ann_index_method!(SSGIndexStr, real_hora::index::ssg_idx::SSGIndex<f32, String>,String);
define_ssg_ann_index!(SSGIndexStr, String);

/// A Python module implemented in Rust.
#[pymodule]
fn hora(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<BruteForceIndexUsize>()?;
    m.add_class::<BruteForceIndexStr>()?;
    // m.add_class::<BPTIndex>()?;
    m.add_class::<HNSWIndexUsize>()?;
    m.add_class::<HNSWIndexStr>()?;
    m.add_class::<PQIndexUsize>()?;
    m.add_class::<PQIndexStr>()?;
    m.add_class::<SSGIndexUsize>()?;
    m.add_class::<SSGIndexStr>()?;
    m.add_class::<IVFPQIndexUsize>()?;
    m.add_class::<IVFPQIndexStr>()?;
    Ok(())
}
