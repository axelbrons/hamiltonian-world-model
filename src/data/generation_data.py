import gymnasium as gym
import numpy as np
import cv2
import os

# Paramètres
NUM_ROLLOUTS = 100  # Commence petit (ex: 100) pour tester, puis monte à 10000 plus tard
MAX_STEPS_PER_ROLLOUT = 300 # Nombre d'étapes max par partie
DATA_DIR = "carracing_dataset"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def process_frame(frame):
    """Redimensionne l'image en 64x64 comme spécifié par David Ha."""
    # Redimensionnement avec OpenCV
    obs_resized = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
    return obs_resized

def generate_data():
    # Utilisation de l'environnement moderne CarRacing-v3
    # render_mode="rgb_array" est crucial pour récupérer les images pixel par pixel
    env = gym.make("CarRacing-v3", continuous=True, render_mode="rgb_array")
    
    print(f"Début de la collecte de {NUM_ROLLOUTS} rollouts...")
    
    for rollout in range(NUM_ROLLOUTS):
        obs, info = env.reset()
        
        frames = []
        actions = []
        rewards = []
        dones = []
        
        # Le jeu a un petit temps de "zoom" au début, on fait quelques pas "à vide" pour passer l'intro
        for _ in range(50):
            obs, _, _, _, _ = env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
            
        for step in range(MAX_STEPS_PER_ROLLOUT):
            # 1. Choisir une action totalement au hasard
            action = env.action_space.sample()
            
            # 2. Appliquer l'action dans l'environnement
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 3. Traiter l'image (redimensionner en 64x64)
            processed_obs = process_frame(obs)
            
            # 4. Sauvegarder dans nos listes temporaires
            frames.append(processed_obs)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            
            obs = next_obs
            
            if done:
                break
                
        # Sauvegarde sur le disque pour ce rollout
        file_name = os.path.join(DATA_DIR, f"rollout_{rollout}.npz")
        np.savez_compressed(
            file_name,
            obs=np.array(frames),
            actions=np.array(actions),
            rewards=np.array(rewards),
            dones=np.array(dones)
        )
        
        if (rollout + 1) % 10 == 0:
            print(f"Rollout {rollout + 1}/{NUM_ROLLOUTS} terminé.")

    env.close()
    print("Collecte terminée !")

if __name__ == "__main__":
    generate_data()