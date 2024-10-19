# Collision Prediction Project

## How to Run the Code

### Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/collision-prediction-project.git
cd collision-prediction-project
```

### Install Dependencies

Ensure you have Python 3.10 and the required libraries installed. To install the dependencies, run:

```bash
pip install -r requirements.txt
```

### Data Collection

(Optional) Run the data collection script to generate your dataset. If you're using the provided data, you can skip this step.

```bash
python Collect_data.py
```

### Training the Model

To train the model, execute the following command:

```bash
python train_model.py
```

### Running the Simulation

To run the goal-seeking simulation, use the command below:

```bash
python goal_seeking.py
```

---

## Dependencies

### Operating System

- Ubuntu 22.04

### Python Version

- Python 3.10

### Required Python Libraries

Below is a list of the required Python libraries specified in the `requirements.txt` file:

```makefile
cython==3.0.3
matplotlib==3.8.0
scikit-learn==1.3.1
scipy==1.11.3
pymunk==5.7.0
pygame==2.5.2
pillow==10.0.1
numpy==1.26.1
noise==1.2.2
torch==2.2.0
torchvision==0.17.0
```

Make sure to have these installed before running the project.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

