# AutoEP

## Introduction
Here are several baselines:
- [eoh](https://github.com/FeiLiu36/EoH)
- [reeov](https://github.com/ai4co/reevo)

## 1. Foundation Building of FastGPT (Agent Framework Building)
[FastGPT](https://github.com/labring/FastGPT) is an efficient knowledge Q&A system based on large language models, supporting private deployment and custom workflow construction.
- **Official GitHub Repository**: [https://github.com/labring/FastGPT](https://github.com/labring/FastGPT)
- **docker-compose**: [https://github.com/labring/FastGPT/blob/main/deploy/docker/docker-compose-oceanbase/docker-compose.yml](https://github.com/labring/FastGPT/blob/main/deploy/docker/docker-compose-oceanbase/docker-compose.yml)

### 1.1 Starting FastGPT with Docker
Execute the `docker-compose up -d` command to start FastGPT.

### 1.2 Connecting Local Models to FastGPT
- **AI Proxy Connection**: You can refer to [https://github.com/labring/FastGPT](https://github.com/labring/FastGPT) to reproduce the code. This is a recommended method. Corresponding to [docker-compose](https://github.com/labring/FastGPT/blob/main/deploy/docker/docker-compose-oceanbase/docker-compose.yml).
- **oneapi Connection**: This method is recommended for long - term use. For the specific connection method, please check [https://doc.tryfastgpt.ai/docs/development/modelconfig/one-api/](https://doc.tryfastgpt.ai/docs/development/modelconfig/one-api/). Corresponding to [docker-compose](https://github.com/MZY199603/AutoEP/edit/main/src/docker-compose.yml).

**Attention!!!**
Compared with connecting to models provided by vendors, it is recommended to connect to privately - deployed large - scale models deployed locally, which can greatly improve experimental efficiency.

### 1.3 API Connection
FastGPT can encapsulate workflows into an API application. For detailed information, please refer to: [https://doc.tryfastgpt.ai/docs/development/openapi/intro/](https://doc.tryfastgpt.ai/docs/development/openapi/intro/).

## 2. Database Deployment

### 2.1 Database Setup
- In this experiment, mysql (8.0.26) is selected for data interaction and storage. The database structure can be viewed in `src/demo.sql`.
- The deployment method can either use docker to pull the image or install and deploy it locally. When using docker for deployment, execute the `docker pull mysql:8.0.26` command to pull the image.

### 2.2 Interactions between the Database and the Algorithm, and between the Database and FastGPT
The database and the FastGPT Agent workflow interact through Flask:
1. **Update the Database**: At the start of each workflow, the utility value of the previous round needs to be updated to the corresponding parameter round.
2. **Database Query**: Provide the hyperparameters and corresponding utility information of 5 rounds for the Agent.
3. **Insert into the Database**: At the end of each workflow, the hyperparameters output by the agent need to be added to the next round.

The interaction between the database and the algorithm mainly involves the output of the large - language model:
4. **Query and Use as Agent Input**: Before each call to the FastGPT workflow, the hyperparameters and utility values of the previous round will be queried and input into FastGPT.

## 3. Example Demonstration (TSP)

### Prerequisite Steps!!!!
The model needs to be configured first. Please refer to Section 1.2 for specific details.

### 3.1 Quick Start Process
1. **Create a New Workflow**: Enter the FastGPT console and click `+ New Workflow` on the right.
![Workflow](src/创建工作流.png "Create Workflow")
2. **Import Configuration**: Use the `workflow_export.json` file provided in this project to import the predefined workflow. After the workflow is imported, replace the model with your configured model, and replace the HTTP module (which corresponds one - to - one with the interface of data_interaction.py) with your local IP address.    
![Import](src/导入1.png "Import Workflow 1")
![Import](src/导入2.png "Import Workflow 2")
3. **Publish the Workflow**: Click the `Publish` button to activate the workflow and record the generated `workflowId` for API calls. At the same time, replace the `key` in the `fastgpt` method of `main.py` with your own key.
![Publish](src/发布.png "Publish")
![Configure API](src/api配置.png "API Configuration")

After completing the above three steps, an agent workflow is successfully published. If the code configuration has not been fully modified, it can be published first and updated later. (Click `Save` and the update is completed after publishing).

### 3.2 Starting the Database and Running the main Code
1. **Database Loading**: If you have a graphical database management software, you can directly import the [demo.sql](https://github.com/MZY199603/AutoEP/edit/main/src/demo.sql) file into mysql. If not, you need to mount the `demo.sql` to the mysql container and run it.
2. **Database Configuration**: After successfully configuring the database, you need to modify the database configuration of `conmysql(self, n)` in `main.py` to your own database configuration, and also make corresponding modifications in `data_interaction`.
3. **Code Execution**:
    - Step 1: Start `data_interaction.py`. Running this file requires installing packages such as Flask and pymysql. After successful startup, you can check the running status of the data interface.
    - Step 2: Start the `main.py` file. Before starting, check:
        - Whether the Http module in the agent workflow matches the port and IP of `data_interaction`.
        - Whether the model is configured successfully. You can first build a simple workflow for testing.
        - Whether the replacement operation in Step 3 of Section 3.1 is completed.
After completing the above three checks, you can run `main.py`.

## 4. OneApi Model Connection (Not required for reproducing this experiment for now. If needed, refer to the following content)

### Deployment via the `src/docker-compose.yaml` Provided in This Article
1. Open the oneapi website. After deploying according to the above `docker-compose`, the access address is the local route address: 3013.
2. Open the "Channels" on the navigation bar, add a new channel, select the custom channel, and fill in the Base_url, channel name, model name, and your KEY. The model name, Base_url, and key need to be obtained from sources such as deepseek and openai and filled in. If it is a locally deployed model, it needs to support the openai interface for invocation.
![Add Channel](src/新增渠道.png "Add Channel")
![Configure Channel](src/添加你的模型渠道.png "Configure Channel")
3. If using oneapi for the first time, click in the Token column to obtain the token and key, and then modify the following two items in the fastgpt configuration of `docker-compose.yaml`:
    - `OPENAI_BASE_URL=http://Local IP or Route IP:3013/v1`
    - `CHAT_API_KEY=The key just obtained` (initially, the quick default key of OneAPI is filled in. After testing is successful, modify it to the newly obtained token key).
4. After adding an LLM, you need to add the added channel in the `config.json` file. An example is as follows:
```json
"llmModels": [{ 
    "model": "gpt-3.5-turbo", // Model name  
    "name": "gpt-3.5-turbo", // Fill in the newly added channel name. The following items can be adjusted as needed and can remain unchanged.  
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
}]