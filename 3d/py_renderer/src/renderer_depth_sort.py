# |=========================================================|
# | Original Code by: Play and Learn to Code                |
# | Depth Sorting by: Degamisu (https://github.com/Degamisu)|
# | Main depth sorting can be found at line #118            |
# |---------------------------------------------------------|
# | Depth sorting is a meathod to cull backfaces so the     |
# | cube does not look transparent. This is important in    |
# | most 3D renderers, to simulate the opacity of an object |
# |---------------------------------------------------------|
# | Please do not remove this text.                         |
# |=========================================================|

import math
import pygame
import numpy as np

# Window settings:
screen_width, screen_height = 800, 600

# Initialize pygame settings:
pygame.init()
pygame.display.set_caption('Python 3D rendering example')
screen = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()
cube_center = {'y': screen_height // 2, 'x': screen_width // 2}
fps = 60

# Colors:
background_color = (240, 235, 240)
cube_color = (255, 0, 0)
circle_color = (0, 0, 255)

# 3D points:
cube_vertices = [
    [-1, -1, 1],
    [1, -1, 1],
    [1, 1, 1],
    [-1, 1, 1],
    [-1, -1, -1],
    [1, -1, -1],
    [1, 1, -1],
    [-1, 1, -1]
]

# Cube sides, each entry is an index in cube_vertices:
cube_sides = [
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [1, 2, 6, 5],
    [0, 3, 7, 4],
    [0, 1, 5, 4],
    [2, 3, 7, 6]
]

angle = 0
angular_change = 0.02
distance = 2
scale = 200
circle_radius = 8
running = True

# Main game loop:
while running:
    clock.tick(fps)
    screen.fill(background_color)
    angle += angular_change

    # Process window events:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Prepare rotation matrices:
    rotation_matrix_x = [
        [1, 0, 0, 0],  # Extra row for compatibility
        [0, math.cos(angle), -math.sin(angle), 0],
        [0, math.sin(angle), math.cos(angle), 0],
        [0, 0, 0, 1]  # Extra row for compatibility
    ]

    rotation_matrix_y = [
        [math.cos(angle), 0, -math.sin(angle), 0],
        [0, 1, 0, 0],
        [math.sin(angle), 0, math.cos(angle), 0],
        [0, 0, 0, 1]
    ]

    rotation_matrix_z = [
        [math.cos(angle), -math.sin(angle), 0, 0],
        [math.sin(angle), math.cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]  # Extra row for compatibility
    ]

    z = 1 / distance

    projection_matrix = [
        [z, 0, 0, 0],
        [0, z, 0, 0],
        [0, 0, z, 0]  # Adjust to match the size of the rotation matrix
    ]

    # Compute the final matrix:
    rotation_matrix = np.matmul(rotation_matrix_y, rotation_matrix_x)  # Correct order of multiplication
    rotation_matrix = np.matmul(rotation_matrix_z, rotation_matrix)
    projection_rotation_matrix = np.matmul(projection_matrix, rotation_matrix)
    projected_vertices = []

    # Project vertices to 2D coordinates:
    for p in cube_vertices:
        # Apply projection matrix to each cube vertex:
        vertex = np.array([p[0], p[1], p[2], 1])  # Extend the vertex to 4 dimensions
        projected = np.matmul(projection_rotation_matrix, np.transpose(vertex))
        print("Shape of projected:", projected.shape)  # Print shape of projected
        x = cube_center['x'] + projected[0] * scale
        y = cube_center['y'] + projected[1] * scale
        projected_vertices.append([x, y, projected[2]])  # Perspective division not needed here

    # Draw vertices as circles:
    for p in projected_vertices:
        pygame.draw.circle(screen, circle_color, (int(p[0]), int(p[1])), circle_radius)

    counter = 1  # Define the counter variable
    # Draw cube sides with depth sorting:
    # Calculate the depth of each cube side:
    depths = []
    for side in cube_sides:
        # Calculate the average depth of the vertices of each side:
        avg_depth = sum(projected_vertices[vertex_index][2] for vertex_index in side) / 4
        depths.append(avg_depth)

    # Sort cube sides based on their depths:
    sorted_sides_indices = sorted(range(len(cube_sides)), key=lambda k: depths[k], reverse=True)

    # Draw cube sides in sorted order:
    for side_index in sorted_sides_indices:
        side = cube_sides[side_index]
        side_coords = [
            (projected_vertices[side[0]][0], projected_vertices[side[0]][1]),
            (projected_vertices[side[1]][0], projected_vertices[side[1]][1]),
            (projected_vertices[side[2]][0], projected_vertices[side[2]][1]),
            (projected_vertices[side[3]][0], projected_vertices[side[3]][1])
        ]
        pygame.draw.polygon(screen, (
            int(cube_color[0] * counter * 0.05), int(cube_color[1] * counter * 0.05),
            int(cube_color[2] * counter * 0.05)), side_coords)
        counter += 3  # Increment counter for shading

    # Present the new image to screen:
    pygame.display.update()

pygame.quit()
