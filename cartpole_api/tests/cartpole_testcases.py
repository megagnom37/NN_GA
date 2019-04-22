import unittest
import sys
import random

sys.path.insert(0, '../src')
import cartpole


class TestNormalUsing(unittest.TestCase):
    class Agent:
        def __init__(self):
            pass

        def predict(self, obs):
            return random.randint(0, 1)

    def setUp(self):
        self.num_agents = 3
        self.agent = [TestNormalUsing.Agent() for _ in range(self.num_agents)]
        self.results = [0 for _ in range(self.num_agents)]

    def test_img_obs(self):
        env = cartpole.CartPole(img_mode=True)
        for i in range(self.num_agents):
            env.prepare_env()
            while not env.is_done():
                obs = env.get_obs()
                action = self.agent[i].predict(obs)
                env.step(action)
            self.results[i] = env.get_reward()
        
        for result in self.results:
            self.assertGreater(result, 0)
    
    def test_discret_obs(self):
        env = cartpole.CartPole()
        for i in range(self.num_agents):
            env.prepare_env()
            while not env.is_done():
                obs = env.get_obs()
                action = self.agent[i].predict(obs)
                env.step(action)
            self.results[i] = env.get_reward()
        
        for result in self.results:
            self.assertGreater(result, 0)

if __name__ == '__main__':
    unittest.main()
