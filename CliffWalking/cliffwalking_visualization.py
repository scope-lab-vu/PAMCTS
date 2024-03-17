import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import matplotlib.image as mpimg

GRIDWIDTH = 12
GRIDHEIGHT = 4
# ^ Create figure and axes
fig, ax = plt.subplots(figsize=(5, 5)) # ^ (5, 5)
# ^ Load images
elf_down_img = mpimg.imread('Images/elf_down.png')
elf_left_img = mpimg.imread('Images/elf_left.png')
elf_right_img = mpimg.imread('Images/elf_right.png')
elf_up_img = mpimg.imread('Images/elf_up.png')
hole_img = mpimg.imread('Images/mountain_cliff.png')
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
    for i in range(GRIDHEIGHT):
        for j in range(GRIDWIDTH):
            ax.add_patch(patches.Rectangle((j, GRIDHEIGHT - 1 - i), 1, 1, linewidth=1, edgecolor='black', facecolor='lightgreen'))

    # ^ Draw holes
    for pos in holes_positions_list:
        hole_extent = [pos[1], pos[1] + 1, GRIDHEIGHT - pos[0] - 1, GRIDHEIGHT - pos[0]]
        ax.imshow(hole_img, extent=hole_extent, aspect='auto', zorder=2)

    # ^ Draw goals
    for pos in goals_positions_list:
        goal_extent = [pos[1], pos[1] + 1, GRIDHEIGHT - pos[0] - 1, GRIDHEIGHT - pos[0]]
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
    img_extent = [agent_current_pos[1], agent_current_pos[1]+1, GRIDHEIGHT - agent_current_pos[0] - 1, GRIDHEIGHT - agent_current_pos[0]]
    ax.imshow(elf_image, extent=img_extent, aspect='auto', zorder=3)


    # * Dashboard text information below the grid
    text_y_position = -0.2  # Y coordinate for text, adjust as needed
    text_spacing = 0.15      # Spacing between lines of text


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
    ax.set_xlim(0, GRIDWIDTH)
    ax.set_ylim(0, GRIDHEIGHT)
    ax.set_xticks(range(GRIDWIDTH))
    ax.set_yticks(range(GRIDHEIGHT))
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
    anim.save(f"cliffwalking_{method_name}_animation.gif", writer='pillow')


if __name__ == '__main__':
    # ^ PA-MCTS
    # method_name = "PAMCTS"
    # agent_positions_list = [[3, 0], [3, 0], [2, 0], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [2, 6], [1, 6], [1, 7], [1, 8], [1, 9], [1, 10], [1, 11], [2, 11], [3, 11]]
    # decisions_list = ["down", "up", "up", "right", "right", "right", "right", "right", "right", "right", "right", "up", "right", "right", "right", "right", "down", "down", "down"]
    # values_list = [0.0, 0.045, 0.096, 0.155, 0.236, 0.32, 0.29, 0.33, 0.37, 0.412, 0.291, 0.673, 0.446, 0.511, 0.546, 0.593, 0.769, 0.907, 0.987]
    # holes_positions_list = [[3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9], [3, 10]]
    # goals_positions_list = [[3, 11]]
    # interval = 600

    # ^ MCTS
    # method_name = "MCTS"
    # agent_positions_list = [[3, 0], [3, 0], [2, 0], [1, 0], [0, 0], [0, 1], [0, 2], [0, 3], [1, 3], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [0, 10], [0, 11], [1, 11], [2, 11], [3, 11]]
    # decisions_list = ["down", "up", "up", "up", "right", "right", "right", "right", "up", "right", "right", "right", "right", "right", "right", "right", "right", "down", "down", "down", "down"]
    # values_list = [0.0, 0.042, 0.089, 0.136, 0.236, 0.373, 0.29, 0.344, 0.223, 0.422, 0.443, 0.479, 0.511, 0.559, 0.614, 0.66, 0.788, 0.893, 0.912, 0.923, 0.94]
    # holes_positions_list = [[3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9], [3, 10]]
    # goals_positions_list = [[3, 11]]
    # interval = 600

    # ^ DDQN
    # method_name = "DDQN"
    # agent_positions_list = [[3, 0], [3, 0], [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [3, 5]]
    # decisions_list = ["down", "up", "right", "right", "right", "right", "right", "right", "right"]
    # values_list = [0.0, 0.13, 0.224, 0.337, 0.326, 0.364, 0.41, 0.422, 0.393]
    # holes_positions_list = [[3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9], [3, 10]]
    # goals_positions_list = [[3, 11]]
    # interval = 600

    # ^ Alphazero
    method_name = "Alphazero"
    agent_positions_list = [[3, 0], [3, 0], [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [3, 7]]
    decisions_list = ["down", "up", "right", "right", "right", "right", "right", "right", "right", "right", "right"]
    values_list = [0.0, 0.144, 0.241, 0.354, 0.291, 0.33, 0.365, 0.29, 0.367, 0.411, 0.43]
    holes_positions_list = [[3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9], [3, 10]]
    goals_positions_list = [[3, 11]]
    interval = 600

    render_method(method_name, agent_positions_list, holes_positions_list, goals_positions_list, decisions_list, values_list, interval)
