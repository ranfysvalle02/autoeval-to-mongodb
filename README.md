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

Of course. Here is a revised version of the blog post, rewritten to be more technical and to directly address the pragmatic realities and limitations of using LLMs for evaluation.

---

## A Pragmatic Framework for Evaluating RAG Systems on MongoDB Atlas

Building a Retrieval-Augmented Generation (RAG) application is no longer the principal challenge; the critical task has shifted to validating its performance. Without a systematic evaluation framework, determining if your RAG system is production-ready is a process of subjective spot-checking and guesswork.

The **autoeval-to-mongodb** project provides an objective, end-to-end blueprint for this validation. It's a runnable toolkit designed to bring quantitative analysis to the complex task of RAG evaluation for systems built on MongoDB Atlas. It addresses the core problem: how do you move from anecdotal evidence to a data-driven report card for your AI?

This project is not just theoretical. It provides a complete, local development environment to build and test a movie-lookup RAG application using:

* **Database:** A local **MongoDB Atlas** instance in a Docker container.
* **Engine:** **MongoDB Atlas Vector Search** for retrieval and **Azure OpenAI** for generation.
* **Core Logic:** A Python-based system orchestrating the RAG pipeline.
* **Validation:** The `autoevals` library for structured, quantitative performance measurement.

### Why the `autoevals` Library?

The evaluation methods in `autoevals` are not novel; they are established industry practices. The library’s value is in addressing the tedious and error-prone mechanics of implementation:
1.  **Normalizing Metrics:** Consistently normalizing scores from different evaluation types (e.g., numeric difference, keyword matching, model-based grades) to a standard 0-to-1 scale is difficult. `autoevals` handles this calculation, ensuring metrics are comparable.
2.  **Parsing Model Outputs:** Debugging model-graded evaluations is challenging. The library provides a robust framework for parsing LLM outputs, managing errors, and iterating on prompts for more reliable results.
3.  **Uniform Interface:** It provides a single, consistent interface for running various evaluation methods, making it simple to swap, combine, and compare different evaluators.

---

## Core Components of the MDBEvalHub Framework

MDBEvalHub is the web-based interface for this framework. It's designed for technical users to structure, execute, and analyze RAG evaluations.

### 1. Automated Test Harness Generation

The first barrier to evaluation is often the setup. MDBEvalHub automates the creation of a baseline testing environment.
* **Atlas Search Indexing:** Directly from the UI, you can target a MongoDB collection, apply a `$match` aggregation for data selection, and generate the required Atlas Vector Search indexes for your specified embedding models.
* **Synthetic Data Generation:** To bootstrap a test suite, the tool can analyze the text content from your selected source field and **automatically generate a set of synthetic Question/Answer pairs**. This creates an initial test dataset, allowing for immediate evaluation without pre-existing ground truth data.

### 2. Flexible Evaluation: Deterministic and Model-Based Checks

A core principle of MDBEvalHub is providing multiple evaluation mechanisms, acknowledging that no single method is sufficient.
* **FunctionEvaluators (Deterministic Math):** For clear-cut, objective criteria, you can use `FunctionEvaluators`. The default implementation is a **regex matcher**, which provides a simple pass/fail score based on whether a specific pattern is found in the output. This is a purely deterministic check with no LLM involvement.
* **LLMClassifiers (Model-Based Judgment):** For more nuanced criteria (e.g., sentiment, style, relevance), you can define an `LLMClassifier`. This uses a large language model as a judge, classifying the output based on a custom prompt and a set of predefined choices with corresponding scores.

### 3. Acknowledging the Flaw: On "LLM as Judge"

Using an LLM for evaluation is an imperfect science. **LLMs can be inconsistent, exhibit bias, and be sensitive to prompt phrasing.** They are not an infallible source of truth.

However, they offer a scalable solution for assessing qualities that are difficult to define with rigid rules. A well-prompted `LLMClassifier` provides a **statistical approximation of human judgment**. The goal is not to achieve a perfect score from a perfect judge, but to establish a **consistent and repeatable baseline**. An upward trend in this imperfect metric during development is a strong signal of improvement. MDBEvalHub treats these scores as valuable signals, not absolute ground truths, to be used alongside deterministic checks. It is a pragmatic step up from having no quantitative signal at all.

### 4. Quantitative Benchmarking and Analysis

When you're ready to run a full test, MDBEvalHub provides a clear, quantitative picture of your system's performance.
* **Side-by-Side Deployment Evaluation:** Run the same test suite against different LLM deployments simultaneously to get direct, comparative data on which models perform best for your use case.
* **Traceable Reporting:** The output is not just a final score. You receive detailed reports with:
    * **Average scores** for each selected metric.
    * **A full breakdown of each test case**, detailing the input, the generated output, the expected output, and the specific context retrieved by Atlas Vector Search.
    * **The complete LLM message history** for debugging prompts and responses.
    * **Performance metrics** like total test duration.

---

### Transform Your RAG Development

MDBEvalHub is designed to shift RAG development from a qualitative art to a data-driven engineering discipline. By combining automated setup, flexible evaluation logic, and detailed, quantitative reporting, it empowers developers to:

* **Accelerate Iteration:** Get fast, objective feedback on changes to prompts, models, or retrieval strategies.
* **Improve Accuracy:** Pinpoint weaknesses by analyzing specific test case failures and the context that led to them.
* **Boost Confidence:** Validate performance against a consistent set of criteria before deploying to production.
* **Optimize Resources:** Make data-informed decisions about which models and configurations provide the best cost-performance ratio.

If you are building RAG applications with MongoDB Atlas and need to move beyond guesswork to confident, data-driven optimization, this framework provides the tools to do so.
