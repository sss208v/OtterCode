---
type: reference
description: openai API 调用方式：endpoint、key、chat completions 参数
importance: 2.5
---
调用 openai 兼容接口的要点：

- base_url 指向内网网关或本地代理
- api_key 用占位值即可，网关层做鉴权
- chat completions 传 model、messages、temperature

超时与重试要有兜底，内网模型推理慢，首 token 延迟可能很高。
