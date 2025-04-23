import cityflow
import numpy as np

class Simulator:
    def __init__(self, intersection_id, phase_id=0, config='config/config.json', action_space_size=5):
        self.eng = cityflow.Engine(config, thread_num=1)       
        self.intersection_id = intersection_id
        self.cur_phase = phase_id
        self.cur_phase_start_time = self.eng.get_current_time()
        self.eng.set_tl_phase(intersection_id, phase_id)
        self.waiting_vehicles = {} # vehicle id -> timestamp started waiting
        self.state = np.array([])
        self.min_phase_duration = 10
    
        self.action_space_size = action_space_size
        
        print(f"Simulator initialized with action_space_size={action_space_size}")
        
    def __update_waiting_vehicles(self):
        '''
        Remove vehicles that left and add vehicles that started waiting.
        '''
        cur_waiting = set()

        for vehicle, speed in self.eng.get_vehicle_speed().items():
            if speed <= 0.1:
                cur_waiting.add(vehicle)

        prev_waiting = set(self.waiting_vehicles.keys())

        for vehicle in prev_waiting - cur_waiting:
                self.waiting_vehicles.pop(vehicle)
        for vehicle in cur_waiting - prev_waiting:
            self.waiting_vehicles[vehicle] = self.eng.get_current_time()

    def __update_state(self):
        '''
        Update the state of the simulation.
        State is defined by a list of lanes, each having the following information:
          The number of vehicles in each lane.
          The number of waiting vehicles in each lane.
          The total wait time of each lane.
        And the current light phase and wait time of each traffic light.
        '''
        self.state = []
        vehicle_count_dict = self.eng.get_lane_vehicle_count()
        wait_count_dict = self.eng.get_lane_waiting_vehicle_count()
        lane_vehicles_dict = self.eng.get_lane_vehicles()

        cur_time = self.eng.get_current_time()
        self.state.append(self.cur_phase)
        self.state.append(cur_time - self.cur_phase_start_time)
        
        for lane in vehicle_count_dict.keys():
            wait_time = 0
            for vehicle in lane_vehicles_dict.get(lane, []):
                wait_time += cur_time - self.waiting_vehicles.get(vehicle, cur_time)
            
            self.state.extend([vehicle_count_dict[lane], wait_count_dict[lane], wait_time])

        self.state = np.array(self.state)

    def get_reward(self):
        '''
        Get the reward of the current state.
        Reward is defined as a combination of wait time (negative) and number of waiting vehicles (negative).
        A flickering penalty is added to discourage rapid phase changes.
        '''
        reward = 0
        
        # Indices in state: [phase, phase_duration, vehicle_count1, waiting_count1, wait_time1, ...]
        for i in range(2, len(self.state), 3):
            # Penalize for vehicles in lane
            vehicle_count = self.state[i]
            reward -= 0.05 * vehicle_count
            
            # Penalize for waiting vehicles
            waiting_count = self.state[i+1]
            reward -= 0.5 * waiting_count
            
            # Penalize for wait time
            wait_time = self.state[i+2]
            reward -= 0.2 * wait_time
        
        # Add penalty for changing phase too quickly (anti-flickering)
        if self.eng.get_current_time() - self.cur_phase_start_time < self.min_phase_duration:
            # Stronger penalty the shorter the phase was
            flickering_penalty = (self.min_phase_duration - (self.eng.get_current_time() - self.cur_phase_start_time)) * 2.0
            reward -= flickering_penalty
        
        return reward

    def step_simulation(self, action=None, step=1, verbose=False):
        '''
        Perform the given action, then step the number of steps and return the state and reward.
        '''
        cur_time = self.eng.get_current_time()

        if action is not None:
            # Check that the phase is changing
            if action != self.cur_phase:
                # Check that the phase change is allowed
                if cur_time - self.cur_phase_start_time >= self.min_phase_duration:
                    # handle enforced right turn if applicable
                    if self.action_space_size == 4:
                        self.eng.set_tl_phase(self.intersection_id, 4)
                        if verbose:
                            print(f"Starting right-turn phase for 5 steps, then will switch to phase {action}")
                        for i in range(5):
                            self.eng.next_step()
                        cur_time = self.eng.get_current_time() # update cur_time b/c we stepped the sim

                    self.eng.set_tl_phase(self.intersection_id, action)
                    self.cur_phase = action
                    if verbose:
                        print(f"Phase changed to {action} after {cur_time - self.cur_phase_start_time} steps")
                        
                    self.cur_phase_start_time = cur_time 
                else:
                    # Phase change not allowed - minimum duration not reached
                    if verbose:
                        print(f"Phase change to {action} denied, minimum duration not reached")
            else:
                # Same phase selected
                if verbose:
                    print(f"Same phase {action} maintained")

        # Step simulation
        for _ in range(step):
            self.eng.next_step()
            self.__update_waiting_vehicles()

        # Update state and calculate reward
        self.__update_state()
        reward = self.get_reward()
        if verbose:
            print(f"Phase: {self.cur_phase}, Duration: {cur_time - self.cur_phase_start_time}")
            print(f"Reward: {reward}")

        return self.state, reward
    
    def reset(self):
        '''
        Reset the simulation to the initial state.
        '''
        self.eng.reset()
        self.cur_phase = 0
        self.cur_phase_start_time = self.eng.get_current_time()
        self.waiting_vehicles = {}
    

if __name__ == '__main__':
    # Test with 5-action space
    print("Testing with 5-action space:")
    sim = Simulator("intersection_1_1", action_space_size=5)
    sim.step_simulation(0, verbose=False, step=50)
    sim.step_simulation(3, verbose=True, step=1)
    sim.step_simulation(verbose=False, step=50)
    
    # Test with 4-action space
    print("\nTesting with 4-action space:")
    sim = Simulator("intersection_1_1", action_space_size=4)
    sim.step_simulation(0, verbose=False, step=50)
    sim.step_simulation(3, verbose=True, step=1)
    sim.step_simulation(verbose=False, step=50)