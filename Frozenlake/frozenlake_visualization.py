import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import matplotlib.image as mpimg


# ^ Constants
GRIDSIZE = 3
FRAME_DURATION = 500  # ^ milliseconds
# ^ Create figure and axes
fig, ax = plt.subplots(figsize=(5, 6)) # ^ (5, 5)
# ^ Load images
elf_down_img = mpimg.imread('Images/elf_down.png')
elf_left_img = mpimg.imread('Images/elf_left.png')
elf_right_img = mpimg.imread('Images/elf_right.png')
elf_up_img = mpimg.imread('Images/elf_up.png')
hole_img = mpimg.imread('Images/hole.png')
goal_img = mpimg.imread('Images/goal.png')


def get_elf_image(agent_current_pos, agent_next_pos, current_decision, elf_images):
    """
    Determine the elf image to use based on the direction of movement.

    :param agent_current_pos: Current position of the agent as [y, x].
    :param agent_next_pos: Next position of the agent as [y, x].
    :param elf_images: Dictionary of elf images with keys 'down', 'left', 'right', 'up'.
    :return: The appropriate elf image.
    """
    if current_decision == "down":
        return elf_images['down']
    elif current_decision == "up":
        return elf_images['up']
    elif current_decision == "right":
        return elf_images['right']
    elif current_decision == "left":
        return elf_images['left']
    else:
        return elf_images['down']


# Function to update the position of the agent in the animation
def update(frame_num, agent_positions_list, holes_positions_list, goals_positions_list, decisions_list, values_list, elf_images):
    ax.clear()  # Clear the previous drawing
    
    # ^ Fill the ground with blue color
    for i in range(GRIDSIZE):
        for j in range(GRIDSIZE):
            ax.add_patch(patches.Rectangle((j, GRIDSIZE - 1 - i), 1, 1, linewidth=1, edgecolor='black', facecolor='skyblue'))

    # ^ Draw holes
    for pos in holes_positions_list:
        hole_extent = [pos[1], pos[1] + 1, GRIDSIZE - pos[0] - 1, GRIDSIZE - pos[0]]
        ax.imshow(hole_img, extent=hole_extent, aspect='auto', zorder=2)

    # ^ Draw goals
    for pos in goals_positions_list:
        goal_extent = [pos[1], pos[1] + 1, GRIDSIZE - pos[0] - 1, GRIDSIZE - pos[0]]
        ax.imshow(goal_img, extent=goal_extent, aspect='auto', zorder=2)

    # * Draw the agent's position with the appropriate elf image
    # * Calculate the agent's position in the grid
    agent_current_pos = agent_positions_list[frame_num]
    # Ensure there's a next position in the list
    if frame_num + 1 < len(agent_positions_list):
        agent_next_pos = agent_positions_list[frame_num + 1]
    else:
        agent_next_pos = agent_current_pos

    
    # * Determine the elf image to use
    current_decision = decisions_list[frame_num]
    elf_image = get_elf_image(agent_current_pos, agent_next_pos, current_decision, elf_images)
    # * Ensure the image extent corresponds to the bottom-left and top-right corners of the correct grid cell
    img_extent = [agent_current_pos[1], agent_current_pos[1]+1, GRIDSIZE - agent_current_pos[0] - 1, GRIDSIZE - agent_current_pos[0]]
    ax.imshow(elf_image, extent=img_extent, aspect='auto', zorder=3)


    # * Dashboard text information below the grid
    text_y_position = -0.1  # Y coordinate for text, adjust as needed
    text_spacing = 0.05      # Spacing between lines of text


    # * Current state text
    current_state_str = f"Current State: {agent_positions_list[frame_num]}"
    ax.text(0, text_y_position, current_state_str, ha='left', transform=ax.transAxes)


    # * Next state text
    if frame_num + 1 < len(agent_positions_list):
        next_state_str = f"Next State: {agent_positions_list[frame_num + 1]}"
    else:
        next_state_str = "Next State: None"
    ax.text(0, text_y_position - text_spacing, next_state_str, ha='left', transform=ax.transAxes)


    # * Decision text
    decision_str = f"Decision: {decisions_list[frame_num]}"
    ax.text(0, text_y_position - 2 * text_spacing, decision_str, ha='left', transform=ax.transAxes)


    # * Node value text
    value_str = f"Decision Q Value Estimate: {values_list[frame_num]}"
    ax.text(0, text_y_position - 3 * text_spacing, value_str, ha='left', transform=ax.transAxes)


    # * Set the grid
    ax.set_xlim(0, GRIDSIZE)
    ax.set_ylim(0, GRIDSIZE)
    ax.set_xticks(range(GRIDSIZE))
    ax.set_yticks(range(GRIDSIZE))
    ax.set_aspect('equal')

    # * Turn off the axes
    plt.axis('off')


def render_method(method_name, agent_positions_list, holes_positions_list, goals_positions_list, decisions_list, values_list, interval):
    """
    render the pamcts gif
    """
    elf_images = {
        'down': elf_down_img,
        'left': elf_left_img,
        'right': elf_right_img,
        'up': elf_up_img
    }

    # Create the animation
    anim = FuncAnimation(fig, update, fargs=(agent_positions_list, holes_positions_list, goals_positions_list, decisions_list, values_list, elf_images), 
                        frames=len(agent_positions_list), interval=interval, repeat=False)

    # save the animation
    anim.save(f"frozenlake_{method_name}_animation.gif", writer='pillow')


def render_initial_grid():
    """
    Renders the initial grid for the frozen lake environment
    """
    agent_positions_list = [[0, 0], [1, 0], [1, 1], [2, 1], [2, 2]]
    holes_positions_list = [[0, 1], [2, 0]]
    goals_positions_list = [[2, 2]]
    elf_images = {
        'down': elf_down_img,
        'left': elf_left_img,
        'right': elf_right_img,
        'up': elf_up_img
    }

    # Create the animation
    anim = FuncAnimation(fig, update, fargs=(agent_positions_list, holes_positions_list, goals_positions_list, elf_images), 
                        frames=len(agent_positions_list), interval=FRAME_DURATION, repeat=False)

    # save the animation
    anim.save('frozenlake_animation.gif', writer='pillow')



if __name__ == '__main__':
    # ^ PA-MCTS
    # method_name = "PAMCTS"
    # agent_positions_list = [[0, 0], [0, 0], [1, 0], [0, 0], [1, 0], [1, 1], [2, 1], [2, 2]]
    # decisions_list = ["down", "left", "left", "up", "left", "up", "down", "right"]
    # values_list = [0.0, 0.872, 0.911, 0.891, 0.886, 0.873, 0.886, 0.989]
    # holes_positions_list = [[0, 1], [2, 0]]
    # goals_positions_list = [[2, 2]]
    # interval = 600

    # ^ MCTS
    # method_name = "MCTS"
    # agent_positions_list = [[0, 0], [0, 0], [1, 0], [0, 0], [1, 0], [1, 1], [1, 2], [0, 2], [1, 2], [2, 2]]
    # decisions_list = ["down", "left", "left", "up", "left", "up", "down", "right", "right", "right"]
    # values_list = [0.0, 0.902, 0.897, 0.921, 0.869, 0.941, 0.944, 0.981, 0.971, 0.988]
    # holes_positions_list = [[0, 1], [2, 0]]
    # goals_positions_list = [[2, 2]]
    # interval = 600

    # ^ DDQN
    # method_name = "DDQN"
    # agent_positions_list = [[0, 0], [0, 0], [1, 0], [0, 0], [0, 1]]
    # decisions_list = ["down", "down", "down", "right", "down"]
    # values_list = [0.0, 0.867, 0.867, 0.903, 0.867]
    # holes_positions_list = [[0, 1], [2, 0]]
    # goals_positions_list = [[2, 2]]
    # interval = 700

    # ^ Alphazero
    method_name = "AlphaZero"
    agent_positions_list = [[0, 0], [0, 0], [1, 0], [0, 0], [0, 1]]
    decisions_list = ["down", "down", "down", "right", "down"]
    values_list = [0.0, 0.834, 0.821, 0.783, 0.8]
    holes_positions_list = [[0, 1], [2, 0]]
    goals_positions_list = [[2, 2]]
    interval = 6000

    render_method(method_name, agent_positions_list, holes_positions_list, goals_positions_list, decisions_list, values_list, interval)
