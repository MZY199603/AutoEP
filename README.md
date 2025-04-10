# AutoEP

##简介  
以下是几个baseline  
[eoh]:(https://github.com/FeiLiu36/EoH)   
[reeohh=]:(https://github.com/ai4co/reevo)   

## 1. FastGPT 简介
[FastGPT](https://github.com/labring/FastGPT) 是一个基于大语言模型的高效知识问答系统，支持私有化部署和自定义工作流搭建。  
🔗 **官方GitHub仓库**: [https://github.com/labring/FastGPT](https://github.com/labring/FastGPT)

### 配置要求
- **大模型配置**：需通过 OneAPI 接入模型服务，配置方法详见 [FastGPT 官方文档](https://doc.tryfastgpt.ai/docs/development/openapi/intro/)。
- **部署文件**：本实验完整部署配置见 `src/docker-compose.yaml`， `src/config.json` 。


## 2. 工作流搭建
### 快速启动流程
1. **新建工作流**  
   - 进入 FastGPT 控制台，点击右侧 `+ 新建工作流`。
2. **导入配置**  
   - 使用本项目提供的 `workflow_export.json` 文件导入预定义工作流。
3. **发布工作流**  
   - 点击 `发布` 按钮激活工作流，记录生成的 `workflowId` 供 API 调用。
  
     
## 3. API 调用与数据存储
### 外部调用接口（与工作流相互调用，保证数据实时给LLM）
通过 HTTP API 触发工作流并存储结果到数据库形成交互：详情见tsp/main.py,与data_interaction.py文件。
本实验，采用的简易mysql（8.0.26）作为数据存储，数据库结构见src/demo.sql   
