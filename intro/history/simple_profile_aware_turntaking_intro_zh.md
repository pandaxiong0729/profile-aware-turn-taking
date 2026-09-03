# Profile-aware Turn-taking：简单 intro

## 1. 我们想解决什么问题

人在自然对话中，并不是只靠“有没有静音”来判断什么时候接话。很多时候，同样一段停顿、短回应或重叠语音，在不同人物关系和对话场景里含义会不一样。

例如：

- 熟人之间的短暂停顿可能是共同思考，不一定表示尴尬；
- 医生和病人、老师和学生、朋友之间，接话和打断的节奏可能不同；
- “嗯”“对”“yeah” 有时只是 backchannel，有时也可能是准备接过话轮。

所以我们的核心问题是：

> 在同一段音频和转写完全不变的情况下，如果模型知道说话人的 profile，它能不能更准确地预测下一步话轮事件？

这里的 profile 指的是说话人的年龄、角色、背景、双方关系和当前场景等信息。

## 2. 我们现在做的任务

当前任务不是让模型生成回复，而是让模型预测下一步对话会发生什么。

我们把原来的五分类话轮事件拆成四个更容易测试的 A/B 问题：

| 二分问题 | A | B |
|---|---|---|
| silence | 没人说话 | 有人说话 |
| listener_onset | 当前说话人继续 | 另一人开始回应 |
| brief_response | 简短反馈 / backchannel | 更实质的接话或打断 |
| yield | 自然换人 | 打断 |

每次测试都比较三种条件：

| 条件 | 含义 |
|---|---|
| hidden | 不给 profile |
| given | 给正确 profile |
| shuffled | 给错误 profile |

公平性控制是：

```text
hidden / given / shuffled 三组里，
音频、转写、样本 ID、预测问题、测试集都不变，
只改变 profile。
```

## 3. 我们怎么把 profile 接到 Qwen 上

我们现在做了两条路线。

### B 路线：Qwen embedding + 外接 adapter

流程是：

```text
因果音频 + 因果转写
        ↓
Qwen context embedding

profile
        ↓
Qwen profile embedding

context embedding + profile embedding
        ↓
shared adapter
        ↓
四个 A/B 输出
```

这条路线更稳定，适合作为当前主结果。

### A 路线：Qwen hidden space + Qwen 自己的 A/B token head

流程是：

```text
因果音频 + 因果转写
        ↓
Qwen hidden vector

profile
        ↓
Qwen profile embedding

hidden vector + profile embedding
        ↓
profile adapter
        ↓
调整后的 Qwen hidden vector
        ↓
Qwen 自己的 A/B token lm_head
        ↓
A/B 概率
```

这条路线更接近“让 Qwen 自己回答 A/B”，但比 B 路线不稳定，所以当前用了 task-specific adapter。

## 4. 当前结果

### B 路线结果

| profile 条件 | silence | listener_onset | brief_response | yield | 平均 |
|---|---:|---:|---:|---:|---:|
| hidden | 0.6200 | 0.6850 | 0.5200 | 0.7800 | 0.6512 |
| given | 0.7080 | 0.7350 | 0.5533 | 0.8000 | 0.6991 |
| shuffled | 0.7000 | 0.7300 | 0.5467 | 0.7900 | 0.6917 |
| given-hidden | +0.0880 | +0.0500 | +0.0333 | +0.0200 | +0.0478 |
| given-shuffled | +0.0080 | +0.0050 | +0.0067 | +0.0100 | +0.0074 |

### A 路线结果

| profile 条件 | silence | listener_onset | brief_response | yield | 平均 |
|---|---:|---:|---:|---:|---:|
| hidden | 0.5040 | 0.5100 | 0.5600 | 0.6600 | 0.5585 |
| given | 0.6800 | 0.6800 | 0.6333 | 0.7400 | 0.6833 |
| shuffled | 0.6760 | 0.6750 | 0.6133 | 0.7300 | 0.6736 |
| given-hidden | +0.1760 | +0.1700 | +0.0733 | +0.0800 | +0.1248 |
| given-shuffled | +0.0040 | +0.0050 | +0.0200 | +0.0100 | +0.0098 |

两条路线都满足当前实验要求：

- 仍然是 embedding；
- 仍然是 A/B 二分问题；
- Accuracy 都高于 50%；
- 测试集一致；
- given 高于 hidden 和 shuffled；
- 接到了 Qwen 模型。

## 5. 这个结果说明什么

当前结果可以支持一个初步判断：

> 在相同音频和相同转写条件下，加入正确 profile 后，模型的话轮预测表现更好。

这说明 profile 不是只作为额外文本装饰，而是可以作为一个可控变量影响 turn-taking 判断。

更重要的是，我们不是只比较“有 profile / 没 profile”，还加入了 `shuffled profile`。如果正确 profile 比错误 profile 更好，说明模型提升不是简单来自“多输入了一段文字”，而更可能和 profile 内容本身有关。

## 6. 目前可以作为的贡献点

当前阶段可以整理成三个贡献点：

1. **提出 profile-aware turn-taking prediction 任务。**  
   不只预测语音中的停顿和重叠，还显式考虑说话人身份、关系和场景。

2. **设计 hidden / given / shuffled 的对照实验。**  
   三组只改变 profile，音频和转写保持一致，用来检验模型是否真的使用 profile。

3. **实现 Qwen-based embedding adapter。**  
   我们验证了两种接入方式：一种是外接 shared adapter，另一种是通过 Qwen 自己的 A/B token head 打分。

## 7. 下一步

下一步可以做三件事：

1. 扩大测试集，确认结果不是小样本偶然现象；
2. 把 profile 从静态信息扩展到动态更新的 relationship / situation；
3. 把 A/B 结果还原到完整的 turn-taking 控制策略，例如继续听、backchannel、接话或打断。

当前最稳的主线建议用 B 路线作为主结果，A 路线作为“更接近 Qwen 原生输出”的补充实验。

