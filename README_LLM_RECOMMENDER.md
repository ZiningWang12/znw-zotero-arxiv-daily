# LLM智能推荐算法

## 概述

新的LLM推荐算法使用Gemini API来智能评估论文的相关性，相比传统的嵌入相似度方法，能够：

1. **更好的语义理解**：理解论文内容与用户研究兴趣的深层关联
2. **个性化推荐**：基于用户历史阅读偏好进行推荐
3. **详细的评分理由**：为每篇论文提供评分解释
4. **更好的区分度**：避免分数扎堆在6-7分的问题
5. **灵活配置**：支持通过YAML文件自定义推荐参数

## 主要特性

### 可配置的研究兴趣领域
默认支持以下研究领域（可通过配置文件自定义）：
- embodied AI
- robotics  
- world model
- multimodal learning
- vision-language models
- robot learning
- autonomous agents
- reinforcement learning
- computer vision
- natural language processing
- robot manipulation
- robot navigation
- imitation learning

### 评分标准
- **9-10分**：与用户核心研究兴趣高度相关，具有重要学术价值
- **7-8分**：与用户研究兴趣相关，值得关注
- **5-6分**：部分相关，可能有一定参考价值
- **3-4分**：相关性较低，但在相关领域
- **1-2分**：基本不相关

## 配置文件设置

在 `private_config.yaml` 中添加 LLM 推荐算法配置：

```yaml
# LLM推荐算法配置
LLM_RECOMMENDER:
  # 研究兴趣领域
  RESEARCH_INTERESTS:
    - "embodied AI"
    - "robotics"
    - "world model" 
    - "multimodal learning"
    - "vision-language models"
    - "robot learning"
    - "autonomous agents"
    - "reinforcement learning"
    - "computer vision"
    - "natural language processing"
    - "robot manipulation"
    - "robot navigation"
    - "imitation learning"
    - "robot grasp"
    - "vision-language-action model"
  
  # 批处理配置
  CORPUS_BATCH_SIZE: 20  # 用于参考的历史论文数量
  CANDIDATE_BATCH_SIZE: 8  # 每批处理的候选论文数量
  
  # 评分配置
  KEYWORD_BONUS: 2.0  # 关键词匹配论文的额外加分
  DEFAULT_SCORE: 5.0  # API调用失败时的默认分数
```

### 配置参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `RESEARCH_INTERESTS` | List[str] | 内置列表 | 用户研究兴趣领域列表 |
| `CORPUS_BATCH_SIZE` | int | 20 | 用于参考的历史论文数量 |
| `CANDIDATE_BATCH_SIZE` | int | 8 | 每批处理的候选论文数量 |
| `KEYWORD_BONUS` | float | 2.0 | 关键词匹配论文的额外加分 |
| `DEFAULT_SCORE` | float | 5.0 | API调用失败时的默认分数 |

## 使用方法

### 1. 配置API密钥

#### 使用Gemini API（推荐）
```bash
export OPENAI_API_KEY="your-gemini-api-key"
export OPENAI_API_BASE="https://generativelanguage.googleapis.com/v1beta/openai/"
export MODEL_NAME="gemini-2.0-flash"
```

#### 使用OpenAI API（备选）
```bash
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_API_BASE="https://api.openai.com/v1"  # 默认值
export MODEL_NAME="gpt-4o"  # 默认值
```

### 2. 启用LLM推荐

在运行主程序时，设置 `--use_llm_api=true`：

```bash
python main.py --use_llm_api=true --zotero_id=your_id --zotero_key=your_key --arxiv_query=cs.AI
```

### 3. 测试功能

运行测试脚本验证功能：
```bash
python test_llm_recommender.py
```

## 算法流程

1. **配置加载**：从YAML文件读取用户自定义的推荐参数
2. **预处理**：根据配置提取对应数量的历史论文作为参考
3. **分批处理**：按配置的批大小将候选论文分批发送给LLM
4. **智能评分**：LLM基于研究兴趣和历史偏好进行评分
5. **关键词加分**：匹配关键词的论文获得可配置的额外加分
6. **排序输出**：按分数降序排列推荐结果

## 性能调优建议

### 批处理大小优化
- **CORPUS_BATCH_SIZE**: 增大可提供更多历史参考，但会增加token消耗
- **CANDIDATE_BATCH_SIZE**: 减小可提高响应速度，但会增加API调用次数

### 成本优化
```yaml
# 低成本配置
LLM_RECOMMENDER:
  CORPUS_BATCH_SIZE: 10    # 减少历史参考论文
  CANDIDATE_BATCH_SIZE: 10 # 增大批处理减少API调用
```

### 精度优化
```yaml
# 高精度配置
LLM_RECOMMENDER:
  CORPUS_BATCH_SIZE: 30    # 更多历史参考
  CANDIDATE_BATCH_SIZE: 5  # 小批处理提高关注度
```

## 容错机制

- **API失败回退**：如果LLM API调用失败，自动回退到传统嵌入相似度方法
- **解析错误处理**：JSON解析失败时使用配置的默认评分
- **批处理重试**：单批次失败不影响其他批次处理
- **配置缺失处理**：缺少配置时使用内置默认值

## 支持的LLM服务

### Gemini API（推荐）
- **优势**：响应速度快，中文理解能力强，成本相对较低
- **配置**：
  ```bash
  export OPENAI_API_KEY="your-gemini-api-key"
  export OPENAI_API_BASE="https://generativelanguage.googleapis.com/v1beta/openai/"
  export MODEL_NAME="gemini-2.0-flash"
  ```

### OpenAI API
- **优势**：稳定性好，API文档完善
- **配置**：
  ```bash
  export OPENAI_API_KEY="your-openai-api-key"
  export OPENAI_API_BASE="https://api.openai.com/v1"
  export MODEL_NAME="gpt-4o"
  ```

## 自定义配置示例

### 专注特定领域
```yaml
LLM_RECOMMENDER:
  RESEARCH_INTERESTS:
    - "embodied AI"
    - "robot manipulation"
    - "world model"
  CORPUS_BATCH_SIZE: 15
  CANDIDATE_BATCH_SIZE: 6
```

### 高通量处理
```yaml
LLM_RECOMMENDER:
  CORPUS_BATCH_SIZE: 10
  CANDIDATE_BATCH_SIZE: 15  # 大批处理
  KEYWORD_BONUS: 1.5        # 减少关键词权重
```

## 注意事项

1. 确保API密钥有足够的配额
2. 大量论文处理时会产生较多API调用费用
3. 网络连接稳定性会影响推荐速度
4. 建议在测试环境中先验证配置是否正确
5. Gemini API在中国大陆地区可能需要配置代理访问
6. 合理设置批处理大小以平衡性能和成本 