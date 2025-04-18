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
   - 点击 `发布` 按钮激活工作流，记录生成的 `workflowId` 供 API 调用。需要把你的key替换掉main.py的fastgpt方法中的key。
  
     
## 3. API 调用与数据存储
### 外部调用接口（与工作流相互调用，保证数据实时给LLM）
1. 通过 HTTP API 触发工作流并存储结果到数据库形成交互：详情见tsp/main.py,与data_interaction.py文件。
本实验，采用的简易mysql（8.0.26）作为数据存储，数据库结构见src/demo.sql   
2. 注意连接数据库时需要配置自己mysql数据库的地址。
3. 在fastgpt中接入data_interaction.py中的flask接口IP改为您自己本机IP.

## 4. OneApi模型接入
### 通过本文提供的`src/docker-compose.yaml`进行部署
1. 打开oneapi网站，按上述docker-compose部署后访问地址为 ---本地路由地址：3013
2. 打开导航条上的渠道，添加新的渠道，选择自定义渠道，填入Base_url，渠道名称，和模型名称和你的KEY。
模型名称和Base_url和key需要你在deepseek，openai等获取后填入，或者是你本地部署的模型，但是要能支持openai的接口方式调用的。 
3. 如果你是首次使用oneapi的话你需要点击令牌列获取你的令牌和key，然后把docker-compose.yaml的fastgpt配置中的进行修改下面两项
 --- OPENAI_BASE_URL=http://本地IP或路由IP:3013/v1
 ---  AI模型的API Key。（开始时默认填写了OneAPI的快速默认key，测试通后，修改为你刚刚获取的令牌key）CHAT_API_KEY=刚刚获得的key 
5. 增加了LLM以后，需要在配置config.json文件中增加添加的渠道，"llmModels": [{
      "model": "gpt-3.5-turbo", //模型名称 
      "name": "gpt-3.5-turbo", //填入你新增的渠道名称，下述其他按照需要进行调整，可以不变。 
      "maxContext": 16000, 
      "avatar": "/imgs/model/openai.svg", 
      "maxResponse": 4000, 
      "quoteMaxToken": 13000, 
      "maxTemperature": 1.2, 
      "charsPointsPrice": 0,
      "censor": false, 
      "vision": false,
      "datasetProcess": true,
      "usedInClassify": true,
      "usedInExtractFields": true,
      "usedInToolCall": true,
      "usedInQueryExtension": true,
      "toolChoice": true,
      "functionCall": true,
      "customCQPrompt": "",
      "customExtractPrompt": "",
      "defaultSystemChatPrompt": "",
      "defaultConfig": {}
    },
...]
   5.配置完成后需要重启服务
   命令 docker-compose down
   docker-compose up-d
   6.workflow中所有LLM换成你配置的LLM即可
## 5. workflow注意事项
   1. **Http协议模块** 
     此处要更换成你的本机的路由IP,并建议单独对改模块进行调试。
   2. **建议**
   先搭好oneapi的配置再进行workflow的搭建。 
     
