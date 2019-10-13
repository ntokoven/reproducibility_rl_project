import matplotlib.pyplot as plt

def plot_time(model_time_map):
    plt.bar(list(model_time_map.keys()), list(model_time_map.values()), width=0.2)
    
    plt.title('Training time')
    plt.ylabel('Duration (in sec)')
    plt.grid(axis='y')
    
    plt.show()

def plot_trials(model_score_map, score_type):
    plt.boxplot(model_score_map.values(), labels=model_score_map.keys())
    
    plt.grid(axis='y')
    plt.title(f'Statistics on {score_type} average return throught {len(list(model_score_map.values())[0])} trials')
    plt.ylabel('Avg return')
    plt.show()

def plot_robustness():
    # std-devs for a fully trained model
    pass

if __name__ == "__main__":
    models = ['A', 'B', 'C']
    times = [100, 200, 300]

    max_returns = [ [10, 20, 30, 20, 15], [50, 60, 70, 45, 55], [30, 60, 90, 120, 80] ]

    model_time_map = dict(zip(models, times))
    model_max_returns_map = dict(zip(models, max_returns))

    plot_trials(model_max_returns_map, 'maximum')
    plot_time(model_time_map)

    # need to save:
    #   time in secs
    #   averge return across trials for each step 
    #   standard error across trials for each step
    #   multiple runs with diff seeds

    # time = int
    # avg ret = list
    # st dev = list
    # algo, seed
    
    
    run_dict = {
        'name' : 'A',
        'avg_ret' : [1 , 2, 4, 3, 3, 4],
        'std_dev' : [0.2, 0.2, 0.4, 0.2, 0.2, 0.5],
        'timestamps' : [1, 3, 5, 6, 7, 9]
    }