# NPL_a7_Training-Distillation-vs-LoRA

In this assignment, we will compare **Odd Layer and Even Layer Student Training Models** with **LoRA (Low-Rank Adaptation)** on a toxic comment classification task using **BERT** from Hugging Face. The goal is to evaluate different model compression techniques while maintaining classification performance.

---

## 📊 Dataset
We use the **OxAISH-AL-LLM/wiki_toxic** dataset from Hugging Face, which contains Wikipedia comments labeled as **toxic (1) or non-toxic (0).**

| Dataset Split | Number of Samples |
|--------------|------------------|
| Train       | 127,656          |
| Validation  | 31,915           |
| Test        | 63,978           |

Each sample contains:
- **id**: Unique identifier
- **comment_text**: The text of the comment
- **label**: (0 = non-toxic, 1 = toxic)

---

## 🛠️ Model Training

### 📌 Odd & Even Layer Distillation
We create a **6-layer student model** from a **12-layer BERT teacher model** by selectively copying either odd or even layers:

#### ✅ Odd Layer Student Model
Trained using teacher layers **{1, 3, 5, 7, 9, 11}**

Student layers mapped as:
```python
student_encoding_layers[i].load_state_dict(teacher_encoding_layers[2*i - 1].state_dict())
```

#### ✅ Even Layer Student Model
Trained using teacher layers **{2, 4, 6, 8, 10, 12}**

Student layers mapped as:
```python
student_encoding_layers[i].load_state_dict(teacher_encoding_layers[2*i].state_dict())
```

#### Model Size Reduction:
- **Teacher Model Parameters**: 109,483,778
- **Student Model Parameters**: 66,956,546 (**61.15% reduction**)

### 📌 LoRA Fine-Tuning
Instead of distilling the model, we use **LoRA (Low-Rank Adaptation)** to fine-tune a full **12-layer BERT** student model with fewer trainable parameters.

#### ✅ LoRA Configuration:
```python
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  
    inference_mode=False,  
    r=8,  
    lora_alpha=32,  
    lora_dropout=0.1  
)
```
- **Trainable Parameters**: 0.27% of the total model (trainable params: 294,912)
- **Total Parameters**: 109M
- **Efficient training with minimal compute requirements**

---

## 📈 Results & Analysis

| Model                     | Training Loss | Test Accuracy |
|--------------------------|---------------|--------------|
| Odd Layer Distillation  | 0.1818        | 93.16%       |
| Even Layer Distillation | 0.1810        | 93.68%       |
| LoRA Fine-Tuning       | 0.2364        | 80.76%       |

### 🔍 Key Takeaways:
✅ **Even Layer Model performed the best (93.68% accuracy)** → Even layers may contain more valuable knowledge.
✅ **Odd Layer Model performed slightly worse (93.16% accuracy).**
✅ **LoRA Model had the lowest accuracy (80.76%)** but required fewer trainable parameters, making it ideal for low-resource settings.

---

## ⚠️ Challenges
- **Distillation Complexity**: Selecting layers is not trivial; even layers performed better than odd layers.
- **LoRA Performance Drop**: While efficient, LoRA struggled with generalization in toxic comment classification.

### 🚀 Proposed Improvements:
- **Hybrid Approach**: Combine distillation with LoRA for better efficiency.
- **Layer-Wise Selection**: Instead of odd/even, experiment with different layer combinations.
- **LoRA Hyperparameter Tuning**: Optimize rank (r), alpha, and dropout for improved accuracy.

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
