# NPL_a7_Training-Distillation-vs-LoRA

In this assignment, we will compare **Odd Layer and Even Layer Student Training Models** with **LoRA (Low-Rank Adaptation)** on a toxic comment classification task using **BERT** from Hugging Face. The goal is to evaluate different model compression techniques while maintaining classification performance.

---

## 📊 Dataset
I chose the dataset OxAISH-AL-LLM/wiki_toxic from Hugging Face. This dataset contains Wikipedia comments labeled as toxic or non-toxic, making it suitable for training and evaluating models on toxic speech detection.

### Data Fields
- **id**: A unique identifier string for each comment.
- **comment_text**: A string containing the text of the comment.
- **label**: An integer, either 0 if the comment is non-toxic, or 1 if the comment is toxic.

### Data Splits
The dataset is divided into three splits: train, validation, and test. Below are the statistics for each split:

| Dataset Split | Number of Data Points |
|--------------|----------------------|
| Train       | 127,656               |
| Validation  | 31,915                |
| Test        | 63,978                |

This dataset will be used to train and evaluate Odd Layer and Even Layer Student Training Models and LoRA (Low-Rank Adaptation) to compare their performance in detecting toxic comments.

---

## 🛠️ Model Training

### 📌 Odd & Even Layer Distillation
In this task, we create a student model from a teacher model by reducing the number of layers from 12 to 6 and then training the student model using either odd or even layers from the teacher model.

#### Creating the Student Model from the Teacher Model
- We first obtain the teacher model's configuration as a dictionary.
- Then, we modify the configuration by reducing the number of hidden layers by half (from 12 to 6).
- Using the modified configuration, we initialize an untrained student model with the same architecture as the teacher model but with fewer layers.

#### Modifying the Student Model Using Even and Odd Layers
To transfer knowledge from the teacher model to the student model, we selectively copy layers:

✅ **Even Layer Student Model** (Layers {2, 4, 6, 8, 10, 12})
- The teacher model has 12 hidden layers.
- The student model has 6 hidden layers.
- We copy every second layer from the teacher to the student model, meaning:
  - Student Layer 1 ← Teacher Layer 2
  - Student Layer 2 ← Teacher Layer 4
  - Student Layer 3 ← Teacher Layer 6
  - Student Layer 4 ← Teacher Layer 8
  - Student Layer 5 ← Teacher Layer 10
  - Student Layer 6 ← Teacher Layer 12

Implementation:
```python
student_encoding_layers[i].load_state_dict(teacher_encoding_layers[2*i].state_dict())
```
This ensures that the student model learns from even-numbered layers of the teacher model.

✅ **Odd Layer Student Model** (Layers {1, 3, 5, 7, 9, 11})
- Instead of using even layers, we now select odd-numbered layers from the teacher model:
  - Student Layer 1 ← Teacher Layer 1
  - Student Layer 2 ← Teacher Layer 3
  - Student Layer 3 ← Teacher Layer 5
  - Student Layer 4 ← Teacher Layer 7
  - Student Layer 5 ← Teacher Layer 9
  - Student Layer 6 ← Teacher Layer 11

Implementation:
```python
student_encoding_layers[i].load_state_dict(teacher_encoding_layers[2*i - 1].state_dict())
```
This modification ensures that the student model learns from odd-numbered layers of the teacher model.

From these two modified models, the student model size is reduced to **61.15% of the teacher model**, with:
- **Teacher parameters:** 109,483,778
- **Student parameters:** 66,956,546

By implementing these modifications, we create two versions of the student model:
1. **Even-layer student model** trained using teacher layers {2, 4, 6, 8, 10, 12}.
2. **Odd-layer student model** trained using teacher layers {1, 3, 5, 7, 9, 11}.


### 📌 LoRA Fine-Tuning
In this task, we apply LoRA (Low-Rank Adaptation) to train a 12-layer student model, reducing the number of trainable parameters while maintaining model performance.

#### 1. Creating the 12-Layer Student Model
By default, the student model is created by reducing the number of layers from 12 to 6, but in this case, we modify the code to keep all 12 layers:

We set:
```python
configuration['num_hidden_layers'] //= 1
```
This ensures that the student model has the same number of layers as the teacher model.

We then copy all layer weights from the teacher model to the student model:
```python
student_encoding_layers[i].load_state_dict(teacher_encoding_layers[i].state_dict())
```
This step ensures that the student model starts with the same initial weights as the teacher.

#### 2. Applying LoRA to Reduce Model Size
After creating the full 12-layer student model, we apply LoRA to reduce the number of trainable parameters.

LoRA Configuration:
```python
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=8, 
    lora_alpha=32, 
    lora_dropout=0.1
)
```

Applying LoRA to the Model:
```python
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
```

#### 3. Results: Trainable Parameters Reduction
```python
trainable params: 294,912 || all params: 109,778,690 || trainable%: 0.2686423020715587
```

---

## 📈 Results & Analysis

### 1. Model Evaluation on the Test Set
We evaluate the models trained using Odd Layers, Even Layers, and LoRA based on their test accuracy. The results are summarized below:

| Model                     | Training Loss | Test Accuracy |
|--------------------------|---------------|--------------|
| Odd Layer Distillation  | 0.1818        | 93.16%       |
| Even Layer Distillation | 0.1810        | 93.68%       |
| LoRA Fine-Tuning       | 0.2364        | 80.76%       |

All three models take approximately the same amount of time to train.

### Performance Analysis
- **Even-Layer Model (Best Performance)**
  - Achieved the highest test accuracy (**93.68%**).
  - Lower training loss, indicating stable and efficient learning.
  - Even layers may capture more informative features, leading to better generalization.

- **Odd-Layer Model (Slightly Lower Performance)**
  - Performed similarly to the Even-Layer model (**93.16% test accuracy**).
  - Higher variability in test accuracy compared to the even-layer model.
  - Suggests that even-numbered layers contribute slightly more to knowledge distillation.

- **LoRA Model (Lower Performance)**
  - Achieved a lower test accuracy (**80.76%**).
  - Training loss is higher, indicating that LoRA adaptation alone may not be sufficient for this task.
  - However, it significantly reduces trainable parameters, making training more computationally efficient.


### 2. Challenges and Comparisons: Distillation vs. LoRA

#### Challenges with Odd/Even Layer Distillation
- **Choosing the Best Layers**: The performance difference between odd and even layers suggests that certain layers carry more valuable knowledge than others.
- **Complexity in Layer Selection**: Unlike full fine-tuning, selecting specific layers for distillation requires careful analysis.
- **Training Stability**: Both models perform well, but slight variations in accuracy indicate that knowledge transfer is not always uniform.

#### Challenges with LoRA Fine-Tuning
- **Lower Test Accuracy**: While LoRA reduces training costs, it performs worse than layer distillation.
- **Limited Adaptation**: Since only a small portion of parameters are updated (**0.27%**), the model may struggle to capture task-specific patterns.
- **Trade-off Between Efficiency and Accuracy**: LoRA is useful for resource-limited scenarios, but it does not match full fine-tuning performance.


### 3. Proposed Improvements
To address these challenges, we can:
1. **Hybrid Approach** – Combine LoRA with selective distillation to improve efficiency without sacrificing performance.
2. **Better Layer Selection** – Instead of using only odd or even layers, analyze which layers contribute most to performance and select the best subset.
3. **LoRA Hyperparameter Tuning** – Adjust LoRA rank (**r**), alpha, and dropout to optimize performance for toxic comment classification.
4. **Layerwise Knowledge Transfer** – Implement progressive distillation, where knowledge is transferred across multiple layers instead of selecting a fixed subset.

---

## 🌍 Web Application

a simple web application that classifies whether a given text input is toxic or hate speech.

### ✅ Features
✔️ **User Input Box** – Users can enter text for classification.
✔️ **Real-Time Prediction** – The model classifies input as toxic or non-toxic.
✔️ **Pretrained Model** – Uses the best fine-tuned model (**Even-Layer Distilled BERT**).

### 🔥 Example:
**Input**: "I hate you"
**Output**: "Toxic"

![1](https://github.com/user-attachments/assets/fa7e9982-567c-45f4-895d-71445f93788f)
![2](https://github.com/user-attachments/assets/dfdd8d6c-f353-4602-aa80-691c066d0450)
![3](https://github.com/user-attachments/assets/da59b691-62b8-4fee-95a1-83f4f266a01a)
![4](https://github.com/user-attachments/assets/218ee753-0b10-4866-99a2-f3d20f5fe083)
![5](https://github.com/user-attachments/assets/22a57aaf-b695-4e76-8904-4cd303b875ac)
![6](https://github.com/user-attachments/assets/81dcffcd-53fd-4221-b55e-072a1aefe9c2)

---

🚀 **This project explores different model compression techniques to optimize BERT for toxic comment classification.**

📌 **Contributions and feedback are welcome!**
