#!/usr/bin/env python
import os, numpy as np, pandas as pd
from loguru import logger

os.makedirs("data", exist_ok=True)
N = 120_000
rng = np.random.default_rng(42)

df = pd.DataFrame({
    "user_id": rng.integers(0, 5000, size=N).astype(str),
    "ad_id": rng.integers(0, 10000, size=N).astype(str),
    "device": rng.choice(["ios","android","web"], size=N, p=[0.4,0.5,0.1]),
    "country": rng.choice(["US","CN","IN","BR","DE","GB"], size=N),
    "age": rng.integers(18, 65, size=N).astype(float),
    "price": rng.lognormal(mean=3.0, sigma=0.7, size=N).astype(float),
})

# 合成点击：逻辑函数 + 若干交互
u_bias = df["user_id"].astype(int) % 7 - 3
d_bias = df["device"].map({"ios":1.0,"android":0.5,"web":-0.3}).values
c_bias = df["country"].map({"US":0.6,"CN":0.4,"IN":0.2,"BR":0.1,"DE":0.3,"GB":0.5}).values
price = np.log1p(df["price"].values)
age = (df["age"].values-18)/47.0

lin =  -2.0 + 0.3*price + 0.4*age + 0.6*d_bias + 0.2*c_bias + 0.05*u_bias
# 显式二阶交叉
lin += 0.15 * price * (df["device"]=="ios").astype(float)
lin += 0.10 * age   * (df["country"]=="US").astype(float)
p = 1/(1+np.exp(-lin))
click = (rng.random(N) < p).astype(int)

df["click"] = click
df.to_csv("data/toy_ctr.csv", index=False)
logger.info("Saved data/toy_ctr.csv with shape={}, CTR={:.3f}", df.shape, click.mean())
