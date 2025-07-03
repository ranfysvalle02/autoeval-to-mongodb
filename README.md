# autoeval-to-mongodb

![](demodemo2.png)

---

### **autoeval-to-mongodb: An End-to-End Blueprint for Building and Validating a RAG System**

This project provides a complete, runnable blueprint for building, testing, and evaluating a Retrieval-Augmented Generation (RAG) application. In today's AI landscape, creating a RAG system is only half the battle; ensuring its reliability and accuracy is paramount. This repository addresses that challenge directly by integrating automated evaluation into the core development workflow.

We demonstrate a practical RAG use case: a movie lookup system that finds film titles based on plot descriptions. The project moves beyond theory and provides a hands-on implementation using a modern, powerful tech stack.

**Key Components:**

* **Database:** A local **MongoDB Atlas** instance running in **Docker**, populated with a sample movie dataset.
* **Intelligence Engine:** **MongoDB Atlas Vector Search** for semantic retrieval, coupled with **Azure OpenAI** for generating embeddings and final responses.
* **Core Logic:** A **Python** script that orchestrates the entire RAG pipeline from user prompt to final answer.
* **Automated Validation:** The **`autoevals` library** is used to systematically measure the system's factuality against a predefined test dataset, producing a quantitative performance score.

The outcome is a tangible demonstration of how to not only build a sophisticated AI system but also to generate a data-driven report card on its performance. This allows developers to iterate, measure improvements, and build more trustworthy AI applications.

### Prerequisites

* **MongoDB Tools:**
  * **mongosh:** The official MongoDB shell for interacting with MongoDB databases.
  * **mongorestore:** A tool for restoring data from a dump file to a MongoDB database.
* **Docker:** Installed on your system ([https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/))
* **wget or curl:** Installed on your system (package managers usually handle this)

### Why does this [autoevals library](https://github.com/braintrustdata/autoevals?tab=readme-ov-file#why-does-this-library-exist) exist?

There is nothing particularly novel about the evaluation methods in the autoevals library. They are all well-known and well-documented. However, there are a few things that are particularly difficult when evaluating in practice:

Normalizing metrics between 0 and 1 is tough. For example, check out the calculation in number.py to see how it's done for numeric differences.
Parsing the outputs on model-graded evaluations is also challenging. There are frameworks that do this, but it's hard to debug one output at a time, propagate errors, and tweak the prompts. Autoevals makes these tasks easy.

Collecting metrics behind a uniform interface makes it easy to swap out evaluation methods and compare them. Prior to Autoevals, we couldn't find an open source library where you can simply pass in input, output, and expected values through a bunch of different evaluation methods.

### Setting Up a Local Atlas Environment

1. **Pull the Docker Image:**

   * **Latest Version:**
     ```bash
     docker pull mongodb/mongodb-atlas-local
     ```

2. **Run the Database:**

   ```bash
   docker run -p 27017:27017 mongodb/mongodb-atlas-local
   ```
   This command runs the Docker image, exposing port 27017 on your machine for connecting to the database.

### Using Sample Datasets with MongoDB

This section demonstrates downloading and exploring a sample dataset for MongoDB on your local system.

#### Downloading the Dataset

There's a complete sample dataset available for MongoDB. Download it using either `wget` or `curl`:

* **Using wget:**

```bash
wget https://atlas-education.s3.amazonaws.com/sampledata.archive
```

* **Using curl:**

```bash
curl https://atlas-education.s3.amazonaws.com/sampledata.archive -o sampledata.archive
```

**Note:**

* Ensure you have `wget` or `curl` installed.
* The downloaded file will be named `sampledata.archive`.

#### Restoring the Dataset

Before restoring, ensure you have a local `mongod` instance running (either existing or newly started). This instance will host the dataset.

**To restore the dataset:**

```bash
mongorestore --archive=sampledata.archive
```

This command uses the `mongorestore` tool to unpack the downloaded archive (`sampledata.archive`) and populate your local `mongod` instance with the sample data.

----

