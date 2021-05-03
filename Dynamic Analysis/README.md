# Dynamic Analysis

To train the model, follow these steps:
cd Dynamic Analysis/Training
python3 parse_sysdig_data.py
python3 convert_to_csv.py
python3 train.py

To run the microservice and the intrusion detection system
cd Dynamic Analysis/vulnerable_web_app
docker compose up -d
cd ..
python3 model.py
