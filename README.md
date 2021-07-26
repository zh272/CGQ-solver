# CGQ: Conjugate Gradient with Quadratic Line Search [[ECMLPKDD2021]](https://2021.ecmlpkdd.org/wp-content/uploads/2021/07/sub_80.pdf)

```bash
@inproceedings{
  hao2021cgq,
  title={Adaptive Learning Rate and Momentum for Training Deep Neural Networks},
  author={Zhiyong Hao and Yixuan Jiang and Huihua Yu and Hsiao-Dong Chiang},
  booktitle={ECMLPKDD},
  year={2021},
}
```

# Dependencies:
numpy, pytorch, matplotlib, sklearn,scipy

# Example usage:
```python
# Compare optimizer performances on same initialization point:
python train_net.py --sgd_epochs=200 --ds_name=cifar10 --batch_size=64 --gpu_id=0 --model=resnet --figure_suffix='test_new/' --seed=6022
```
