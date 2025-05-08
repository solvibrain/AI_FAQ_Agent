---
trigger: glob
---

For each component creating one file with naming convention signify its application and after implementing and testing each component there will be main file which will be integration of all component.
Maintaining Modular Approach through out the project.
Maintaing utils file in which implementation of all functions or classes will be there and we will import function from this file.
Maintaing one seperate file for using langchain utilities implemented function like document-loader, TextSpliting,Retriever, LLMProvider, Vector DB. And using functions implemented in this file into utils as well as main file.
Maintaing one folder for Data Repository in which we will store data files.
Maintaing one file for using gradio and running gradio application from there which will use the final main integrated file to render backeend logic.
