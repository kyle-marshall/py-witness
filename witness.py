#!/usr/bin/env

import pygame
from pygame import *

import math
import random
import numpy as np
PI = math.pi

def softmax(x):
    e_x = np.exp(x-np.max(x))
    return e_x / e_x.sum()

class Poly:
    def __init__(self, points):
        self.points = points
        

class PuzzleToken:
    # the symbols that clue to how the puzzles work
    def __init__(self):
        pass
"""
class PuzzleNode():
    def __init__(self):0
        self.connected = []
        self.is_start = False
        self.is_end = False"""

class HalfEdge:
    def __init__(self, origin, twin, incident_face):
        """
            origin: the origin Vertex
            twin: the HalfEdge on the 'right' side that completes the edge
            incident_face: face to the left
        """
        self.origin = origin
        # twin is the edge on the other side of the incident_face
        self.twin = twin

        # the 'inner' face / the face to this half edge's left
        # (h-edges go counter clockwise around face)
        self.incident_face = incident_face
        # incident_face can be None along edge
        
        # next half-edge starts at twin.origin, continues ctr-clockwise to next vert
        self.next = None
        # some implemenetations only store a next
        self.prev = None
        # to allow the puzzle graph to have stray edges
        # a degenerate will have the same face on either side
        # i.e. incident_face = twin.incident_face
        self.is_degenerate = False
        
    def set_next(self, half_edge):
        self.next = half_edge
    def set_prev(self, half_edge):
        self.prev = half_edge
    def get_dest(self):
        return self.twin.origin
        
class Vertex:
    def __init__(self, coord, puzzle_node):
        """
            coord: a tuple (x, y)
        """
        self.coord = coord
        # half edge which has this vertex as its origin
        self.incident_edge = None
    def set_incident_edge(self, edge):
        self.incident_edge = edge

class Face:
    def __init__(self, face, some_half_edge):
        # can be any one of the half-edges that form the face
        self.incident_edge = some_half_edge
        
# based on a "double connected edge list"
class PuzzleGraph():
    # we're all just made of half edges, in the end
    def __init__(self):
        self.half_edges = []
        self.vertices = []
        self.faces = []
        self.valid_bounds = False
        self._bounds = None
    def get_bounding_rect(self):
        if self.valid_bounds:
            return self._bounds
        min_x, max_x = None, None
        min_y, max_y = None, None
        for v in vertices:
            x, y = v.coord
            if min_x is None or x < min_x:
                min_x = x
            if max_x is None or x > max_x:
                max_x = x
            if min_y is None or y < min_y:
                min_y = y
            if max_y is None or y > max_y:
                max_y = y
        w = max_x - min_x
        h = max_y - min_y
        return (min_x, min_y, w, h)
    
    def draw_to_fit(self, surf):
        start_x, start_y, gw, gh = self.get_bounding_rect()
        sw, sh = surf.get_size()
        # iterate through the
        
# base class for different kinds of puzzles
class Puzzle():
    def __init__(self, grid_size):
        self.grid_size = grid_size;
        self.node_grid = []
        # solution will be a sequence of node indices
        self.solution = []
        # this is a grid that is 1 smaller than the node grid in both dimensions
        # each item represents a cell that may have a token in it or None
        self.token_grid = []
        self.init_grid()

    def init_grid(self):
        n_w, n_h = self.grid_size
        t_w, t_h = (v_w - 1, v_h - 1)
        
        self.node_grid = []
        self.token_grid = []
        
        for x in range(n_w):
            col = []
            for y in range(n_h):
                col.append(PuzzleNode())
            self.node_grid.append(col)
            
        for x in range(t_w):
            col = []
            for y in range(t_h):
                col.append(None)
            self.token_grid.append(col)

    def get_token(self, grid_pos):
        t_x, t_y = grid_pos
        token = self.token_grid[t_x][t_y]
        return token
    def get_node(self, grid_pos):
        n_x, n_y = grid_pos
        node = self.node_grid[n_x][n_y]
    def set_token(self, grid_pos, token):
        t_x, t_y = grid_pos
        self.token_grid[t_x][t_y] = token
    def set_node(self, grid_pos, node):
        n_x, n_y = grid_pos
        self.node_gird[n_x][n_y] = node
    
        
class PuzzleFactory():
    def __init__(self):
        pass
    def MakeBlackWhitePuzzle(self, grid_size):
        """
            generates a random black/white grouping puzzle
            grid_size: a tuple (wide, tall) in terms of vertex count
        """
        bw = Puzzle(grid_size)
        n_w, n_h = grid_size
        # the number of tokens wide/tall is one less than number of nodes
        t_w, t_h = (n_w-1,n_h-1)
        # keep generating random puzzles till we get one that is...
        #  -solvable  -not
        satisfied = False
        cell_count = g_w * g_h
        while not satisfied:
            # attempt to make a great puzzle
            # clear cells
            bw.init_grid()
            # use random weights to choose number of cells
            board_fill_rat = random.random()
            b_rat, w_rat = softmax([random.random(), random.random()])
            # after softmax, b_rat and w_rat add up to 1.0
            b_rat * board_fill_rat
            w_rat *= board_fill_rat
            try_count = 10
            # try try_count times to place the target amount of cells before trying new settings instead
            while try_count > 0 and not satisfied:
                # place cells
                
                # is it solvable?
                if solvable:
                    satisfied = True
                try_count -= 1
                
def main():
    pygame.init()
    screen = pygame.display.set_mode((600,600))
    screen.fill((0,0,0))
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == QUIT:
                running = False
            elif e.type == KEYDOWN:
                if e.key == K_ESCAPE:
                    running = False
        screen.fill((0,0,0))
        
        #pygame.display.update()
    pygame.quit()
if __name__ == "__main__":
    main()
