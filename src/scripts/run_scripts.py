import os

for rain_event in range(1, 16):
    cmd = f"python test_trained_model.py {rain_event} reward_244_episodes_3000 {True}"
    os.system(cmd)