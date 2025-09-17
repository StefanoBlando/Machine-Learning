# TIM Recommender System

This project focuses on building a predictive recommender system to maximize the effectiveness of marketing campaigns by suggesting the most relevant actions to customers. Developed as part of the **CESMA Master's program** at the University of Rome "Tor Vergata" in collaboration with **TIM**, this solution employs a robust machine learning pipeline.

The repository is structured to organize the project's code into modular components, reflecting the logical flow established across the provided Jupyter notebooks.

---

### Repository Structure

/tim-recommender-system
├── notebooks/
│   └── TIM ML LAB 1.ipynb
├── src/
│   ├── data_loader.py
│   ├── models.py
│   ├── feature_enhancements.py
│   ├── hyperparameter_tuning.py
│   ├── ensembles.py
│   └── main.py
├── visualizations/
└── README.md

-   **`notebooks/`**: Contains the original Jupyter Notebooks for reference and step-by-step analysis.
-   **`src/`**: Holds the Python modules, where each file corresponds to a logical stage of the project. This structure promotes code reusability and maintainability.
-   **`visualizations/`**: A folder to store all the generated plots and dashboards, offering a visual summary of the project's results.

---

### Methodology

The solution is built on a principled and rigorous methodology, progressing from initial data analysis to advanced ensemble modeling.

1.  **Data Loading & Preparation (`src/data_loader.py`)**
    The `TIMDataLoader` class handles loading the raw data, performing validation, and transforming it into a **Learning-to-Rank (LTR)** format suitable for the modeling phase. This ensures that the problem is correctly framed as a ranking task.

2.  **Model & Training (`src/models.py`)**
    This module contains the logic for training baseline tree-based models, such as **LightGBM** and **XGBoost**. It uses **Group K-Fold Cross-Validation** to prevent data leakage and provide a reliable performance assessment by ensuring a customer's data is contained within a single fold.

3.  **Feature Engineering & Enhancement (`src/feature_enhancements.py`)**
    Building upon the basic temporal and PCA features, this module introduces advanced features derived from customer and action profiles, as well as historical patterns. The implementation is carefully designed to avoid data leakage, ensuring the model's validity on unseen data.

4.  **Hyperparameter Tuning (`src/hyperparameter_tuning.py`)**
    This module employs **Bayesian Optimization with pruning** to efficiently find the optimal set of hyperparameters for the models. The tuning process rigorously evaluates model performance and stability, a key factor in producing a reliable final solution.

5.  **Advanced Ensemble Methods (`src/ensembles.py`)**
    The final stage combines the predictions of the best-performing models to achieve an additional performance boost. This module implements several advanced ensemble strategies:
    -   **Weighted Average**: Simple linear combination of model predictions.
    -   **Learned Blending**: Uses a regression model to learn optimal weights.
    -   **Stacked Ensemble**: A meta-learner uses base model predictions as new features.
    -   **Ranking-Aware**: Weights models based on their top-k ranking performance.
    -   **Confidence-Weighted**: Uses prediction variance to determine model weights.
    -   **Multi-Level**: A hierarchical approach combining multiple ensemble types.

---

### Key Results

The project successfully achieved a significant improvement over the baseline model, demonstrating the effectiveness of the advanced machine learning pipeline.

| Stage                   | NDCG@5 Score | Improvement vs Baseline |
| :---------------------- | :----------- | :---------------------- |
| **Baseline Model** | 0.5030       | --                      |
| **Best Single Model** | 0.6838       | +35.94%                 |
| **Best Ensemble** | 0.6852       | +36.23%                 |

This structured approach not only delivers strong predictive performance but also provides a robust and validated solution ready for production or further development.

---

### How to Use the Code

1.  **Repository Setup**: Clone this repository to your local machine.
2.  **Data Placement**: Create a folder named `data/` in the root directory and place your `actions.csv` and `features.csv` files inside it.
3.  **Code Migration**: Copy the code from the respective sections of your original Jupyter Notebook into the corresponding `.py` files in the `src/` directory.
4.  **Execution**: You can run each module sequentially by calling its `main()` function, ensuring that the output from one module (e.g., `train_df`, `test_df`) is available for the next.
