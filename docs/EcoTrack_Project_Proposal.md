# FAST — NATIONAL UNIVERSITY OF COMPUTER AND EMERGING SCIENCES

## FAST School of Computing — Department of Artificial Intelligence & Data Science

# ECO TRACK: HOUSEHOLD ENERGY CONSUMPTION & COST ANALYZER

**Project Proposal**

**Course:** AL-2001 — Programming for Artificial Intelligence (Laboratory)  
**Semester:** Fall 2026  
**Section:** BAI-3A  
**Group Members:**  
- **Syed Abdullah Shah (25K-0142)** — Group Leader
- **Bhawani Shanker (25K-0044)**

## 1. Introduction and Problem Statement

Household electricity consumption is an important concern because inefficient energy usage can increase electricity bills and waste valuable resources. Electricity readings can be difficult for ordinary users to interpret, especially when they want to identify consumption patterns, estimate costs, compare usage across time periods, or understand which areas may require attention. Raw electricity records alone do not provide clear summaries, visual explanations, or practical recommendations.

EcoTrack is a Python-based, menu-driven console application designed to analyze household electricity consumption, estimate electricity costs, visualize usage patterns, and provide energy-saving recommendations. The project will combine programming fundamentals, object-oriented programming, file handling, numerical computation, data analysis, visualization, introductory machine learning, text processing, and public API integration in one modular application.

## 2. Proposed Solution and Main Features

EcoTrack will provide the following features:

1. **Energy data management:** Read, validate, organize, and store household electricity records using CSV, JSON, and TXT files.
2. **Consumption analysis:** Calculate hourly, daily, and monthly consumption summaries and identify high-usage periods.
3. **Cost estimation:** Estimate electricity costs using calculated energy consumption and a user-provided tariff per kilowatt-hour (kWh).
4. **Numerical and tabular analysis:** Use NumPy for vectorized calculations and Pandas for data cleaning, filtering, grouping, and summarization.
5. **Data visualization:** Generate at least three properly labelled charts, such as consumption over time, daily or monthly usage comparisons, and usage distributions.
6. **Machine learning:** Compare Logistic Regression and Decision Tree models using the same chronological train/test split to classify electricity-consumption patterns. Evaluation will use held-out test data. **Label assignment:** before any model is trained, each hourly record will be labelled by comparing its energy consumption (kWh) with a threshold calculated from the training portion only (for example, the training-set median). Records above the threshold are labelled high consumption and the rest low consumption. The same threshold is applied to the test data, and the models predict the next hour's label from earlier consumption and time-based features, without using future information.
7. **Text processing:** Clean and normalize energy-related notes using custom Python code, then transform the text into TF-IDF features for exploratory analysis or classification where appropriate.
8. **Energy-saving recommendations:** Provide rule-based suggestions based on observed consumption patterns and user-provided appliance information.
9. **Weather information:** Retrieve weather data through the Open-Meteo public API using the Requests library. The application will check HTTP status codes, cache successful responses locally in JSON, and use cached data when the network is unavailable.
10. **Reliable console interaction:** Provide a menu-driven interface with input validation and clear error messages for invalid records, missing files, and unavailable network services.

The application will remain a console-based prototype. It will not include a graphical or web interface, database server, deep learning, or deployment.

## 3. Dataset, Data Handling, and Tools

**Primary dataset:** UCI Individual Household Electric Power Consumption Dataset  
https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

**Public API:** Open-Meteo Weather API  
https://open-meteo.com/en/docs

A small, documented subset of the UCI dataset covering a limited date range will be stored in `data/raw/` to keep the repository manageable and ensure the project can run without downloading the full dataset. Processed records will be stored separately in `data/processed/`. Results and limitations will be described in relation to the selected subset.

The application will validate records, handle missing or malformed values, and use CSV, JSON, and TXT files for appropriate data-storage tasks. Electricity cost estimates will be calculated from energy measured in kWh rather than treating instantaneous power measurements as energy. For varying readings, the program will use an appropriate time-based aggregation method.

The weather feature will accept a location through a documented input method. If users enter a place name, a geocoding step will resolve it to coordinates before requesting weather data. Weather information will provide context and will not be treated as proof that weather caused changes in historical electricity consumption.

**Technologies:** Python 3, NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, Requests, CSV, JSON, TXT, Git, GitHub, and VS Code.

## 4. Object-Oriented Design and Laboratory Coverage

The project will use a modular object-oriented design with at least seven planned classes:

| Class | Responsibility |
|---|---|
| `EnergyDataManager` | Manage electricity records and file operations. |
| `EnergyAnalyzer` | Perform numerical calculations and consumption analysis. |
| `CostCalculator` | Estimate electricity costs using energy consumption and user-provided rates. |
| `EnergyRecommendationStrategy` | Abstract base class defining the recommendation interface. |
| `UsageBasedRecommendation` | Generate recommendations from consumption patterns. |
| `ApplianceBasedRecommendation` | Generate recommendations using user-provided appliance information. |
| `WeatherService` | Retrieve, validate, and cache weather information. |

The recommendation classes will demonstrate inheritance, method overriding, and polymorphism. Constructors and encapsulation will be used throughout the object-oriented core.

The project will cover the course requirements as follows:

| Lab | Planned implementation |
|---|---|
| 01–02 | Variables, operators, type conversion, validated input, conditions, loops, and suitable comprehensions. |
| 03 | Lists, tuples, dictionaries, and sets, selected according to their intended uses and justified in the report. |
| 04 | Modular functions with parameters, default arguments, return values, and docstrings; at least one lambda used with `sorted(key=...)`, `filter()`, or `map()`. |
| 05 | At least three classes with constructors and encapsulation, inheritance, overriding, polymorphism, and an abstract base class. |
| 06 | Reading and writing at least two of TXT, CSV, and JSON; a custom exception class; specific exception handling; Requests-based API access, status checking, and offline caching. |
| 07 | A genuine two-dimensional NumPy computation using slicing or boolean masking, broadcasting, and axis aggregation, fully vectorized without Python loops over array elements. |
| 08 | A Pandas DataFrame built from project data, cleaned of missing values, filtered, grouped, and summarized. |
| 09 | At least three labelled Matplotlib or Seaborn visualizations, each interpreted in the final report. |
| 10–11 | A scikit-learn train/test split, fitted model, evaluation on unseen data, and comparison of Logistic Regression and Decision Tree on the same split. |
| 12 — Track A | Custom text cleaning and normalization followed by TF-IDF feature extraction. |
| 14 | Modular source files, docstrings throughout, README, pinned `requirements.txt`, `.gitignore`, and ethical considerations. |

The final report will identify where each required concept is implemented in the source code.

## 5. Division of Work

The project will be developed by two members. Its scope is intentionally limited to a console-based prototype, a small dataset subset, two compared machine-learning models, one text-processing track, and one public weather API to keep the workload manageable for a two-member group.

| Member | Responsibilities |
|---|---|
| **Syed Abdullah Shah (25K-0142), Group Leader** | Main application structure, energy data management, object-oriented design, file handling, input validation, custom exceptions, NumPy and Pandas analysis, and cost estimation. |
| **Bhawani Shanker (25K-0044)** | Data visualizations, machine-learning models and evaluation, TF-IDF text processing, energy recommendation functionality, and Open-Meteo API integration with offline caching. |

Both members will participate in integration, testing, documentation, and final preparation. Each member will make meaningful commits through their own GitHub account. The final report will accurately identify the files, functions, and components contributed by each member.

## 6. Development Milestones

| Milestone | Planned work |
|---|---|
| 1 | Finalize requirements, select the dataset subset, and establish the GitHub repository structure. |
| 2 | Implement data management, object-oriented classes, file handling, validation, and exception handling. |
| 3 | Implement NumPy computations and Pandas data cleaning and analysis. |
| 4 | Implement cost estimation and at least three visualizations. |
| 5 | Train, evaluate, and compare the two machine-learning models using the same chronological split. |
| 6 | Implement custom text cleaning, TF-IDF features, and energy-saving recommendations. |
| 7 | Integrate the weather API, coordinate resolution, response checking, and offline cache. |
| 8 | Integrate modules, test error cases, verify clean-clone execution, and prepare the final documentation. |

## 7. Expected Outcome, Ethics, and Limitations

The expected outcome is a working Python console application that transforms household electricity records into understandable analytical summaries, estimated electricity costs, visualizations, consumption classifications, and practical energy-saving recommendations. It will demonstrate the integration of the programming concepts and libraries taught in AL-2001.

Household electricity data can reveal patterns in people's daily activities. The project will avoid publishing private household records or personal information and will follow the usage conditions of its datasets and external API. API responses will be handled responsibly, and credentials or secrets will not be committed to the repository.

The selected dataset subset may not represent every household, region, or climate. Machine-learning results may be limited by dataset size, class distribution, and feature quality. Recommendations and cost calculations will be estimates rather than guarantees of savings. Weather information will be treated as contextual data rather than evidence of causation.

## 8. Repository and Documentation Plan

The public GitHub repository will follow the structure required by the course guidelines:

- `README.md` — project description, members and contributions, setup instructions, exact run command, features, limitations, and data-source links.
- `requirements.txt` — pinned dependencies.
- `.gitignore` — excludes caches, virtual environments, and secrets.
- `docs/` — proposal and final report in both PDF and Markdown formats.
- `src/` — modular Python source files.
- `data/raw/` and `data/processed/` — raw subset and processed data.
- `outputs/figures/` and `outputs/results/` — generated charts and analysis results.

The application will be tested from a clean repository clone on another machine using the README instructions. The final report will document the actual implementation, dataset, results, testing, individual contributions, and any external code, datasets, or AI assistance used.

**Project Deliverable:** A modular, menu-driven Python application integrating household energy analysis with the required AL-2001 programming concepts, data-science libraries, text processing, machine learning, and API handling.
