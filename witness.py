#!/usr/bin/env python3

import pygame
from pygame import *

import math
import random
import numpy as np

def softmax(x):
    e_x = np.exp(x-np.max(x))
    return e_x / e_x.sum()

def avg_point(point_list):
    x_sum = sum([x for x,y in point_list])
    y_sum = sum([y for x,y in point_list])
    n = len(point_list)
    return (x_sum / n, y_sum / n)

class Poly:
    def __init__(self, points):
        self.points = points
        
class Colors:
    RED = (255, 0, 0)
    ORANGE = (255, 180, 0)
    YELLOW = (255, 255, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    BLACK = (0,0,0)
    WHITE = (255,255,255)
    def __init__(self):
        pass
        
class PuzzleToken:
    # types
    COLOR_BLOCK = 0 # for the black-white or multi-color seperation puzzles
    FIXED_SHAPE = 1 # tetris-like pieces that can't be rotated
    FREE_SHAPE = 2 # "  "  "  that can be rotated
    FIXED_NEG_SHAPE = 3 # the blue subtraction shapes
    FREE_NEG_SHAPE = 4 # " 
    COLOR_STAR = 3 #  they must come in twos

    # custom property name constants
    PROP_COLOR = "color"
    PROP_SHAPE = "shape"
    
    # the symbols that clue to how the puzzles work
    def __init__(self, token_type, properties = {}):
        self.type = token_type
        self.properties = properties
    def draw(self, surf, pos, rad):
        t = self.type
        if t == COLOR_BLOCK:
            col = properties[PROP_COLOR]
            x,y = pos
            rect = (x-rad,y-rad,rad*2,rad*2)
            pygame.draw.rect(surf, col, rect)
        # TODO: draw other types

class TokenFactory:
    def __init__(self):
        pass
    @classmethod
    def make_color_block_token(self, color):
        col_key = PuzzleToken.PROP_COLOR
        t = PuzzleToken.COLOR_BLOCK
        return PuzzleToken(t, { col_key : color } )

"""
class PuzzleNode():
    def __init__(self):0
        self.connected = []
        self.is_start = False
        self.is_end = False"""

# Puzzle nodes are stored as the payloads of the below defined DCEL Vertex
class PuzzleNode():
    def __init__(self):
        self.puzzle_node = None
        self.is_start = False
        self.is_end = False
        self.token = None # sometimes there is a special dot on vertices, could be colored

# DCEL objects
class HalfEdge:
    def __init__(self, origin):
        """
            origin: the origin Vertex
            twin: the HalfEdge on the 'right' side that completes the edge
            incident_face: face to the left
        """
        self.origin = origin
        # twin is the edge on the other side of the incident_face
        self.twin = None

        # the 'inner' face / the face to this half edge's left
        # (h-edges go counter clockwise around face)
        self.incident_face = None
        # incident_face can be None along edge
        # next half-edge starts at twin.origin, continues ctr-clockwise to next vert
        self.next = None
        # some implemenetations only store a next
        self.prev = None
        
        # FOR WITNESS SIM
        self.is_degenerate = False
        # can player go through this edge
        self.is_passable = True
    def set_twin(self, twin):
        self.twin = twin
        # it goes both ways!
        twin.twin = self
    def set_incident_face(self, face):
        self.incident_face = face
    def set_next(self, half_edge):
        self.next = half_edge
    def set_prev(self, half_edge):
        self.prev = half_edge
    def get_dest(self):
        return self.twin.origin
        
class Vertex:
    def __init__(self, coord):
        """
            coord: a tuple (x, y)
        """
        self.coord = coord
        # half edge which has this vertex as its origin
        self.incident_edge = None

        
        # TO STORE DATA FOR THE WITNESS
        self.puzzle_node = None
        
    def set_incident_edge(self, edge):
        self.incident_edge = edge
    def get_puzzle_node(self):
        return self.puzzle_node
    def set_puzzle_node(self, node):
        self.puzzle_node = node


class Face:
    def __init__(self, some_half_edge = None):
        # can be any one of the half-edges that form the face
        self.incident_edge = some_half_edge
        self.color = None
        self.token = None
    def set_incident_edge(self, some_half_edge):
        self.incident_edge = some_half_edge
    def get_token(self):
        return token
    def set_token(self, token):
        self.token = token
# based on a "double connected edge list"
class PuzzleGraph():
    # we're all just made of half edges, in the end
    def __init__(self, vertices = [], half_edges = [], faces = []):
        self.half_edges = half_edges
        self.vertices = vertices
        self.faces = faces
        self.valid_bounds = False
        self._bounds = None
    def get_bounding_rect(self):
        if self.valid_bounds:
            return self._bounds
        min_x, max_x = None, None
        min_y, max_y = None, None
        for v in self.vertices:
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
        x_off, y_off, gw, gh = self.get_bounding_rect()
        sw, sh = surf.get_size()
        # source is graph, dest is surf
        scale_x = float(sw) / gw
        scale_y = float(sh) / gh
        # draw faces
        for f_i, face in enumerate(self.faces):
            # get all the points of the face
            start_edge = face.incident_edge
            next_edge = start_edge.next
            coords = [start_edge.origin.coord]
            while next_edge is not None and next_edge is not start_edge:
                coords.append(next_edge.origin.coord)
                next_edge = next_edge.next
            # transform to surface dimensions
            pts = [ ((x-x_off)*scale_x, (y-y_off)*scale_y)
                    for x,y in coords ]
            
            if face.color is not None:
                col = face.color
            else:
                col = tuple([random.randint(0,255) for i in range(3)])
            # draw a poly
            if len(pts) > 2:
                pygame.draw.polygon(surf, col, pts)
                # draw the token, too
                if face.token is not None:
                    face.token.draw(surf, avg_point(pts), 10)
            else:
                print("GRAPH ERROR AROUND FACE #%s!"%f_i)
                print(pts)
        # draw edges
        for edge in self.half_edges:
            xi, yi = edge.origin.coord
            xf, yf = edge.twin.origin.coord
            start = ((xi-x_off)*scale_x, (yi-y_off)*scale_y)
            end = ((xf-x_off)*scale_x, (yf-y_off)*scale_y)
            col = (20, 20, 20)
            thick = 3
            pygame.draw.line(surf, col, start, end, 5)
        for v in self.vertices:
            x,y = v.coord
            pt = (int((x-x_off)*scale_x), int((y-y_off)*scale_y))
            if v.puzzle_node is not None:
                if v.puzzle_node.is_start:
                    pygame.draw.circle(surf, col, pt, 10) 

def make_test_graph():
    graph = PuzzleGraph()
    # this example shows that DCELs are kind of a pain to build by hand
    # create a connected grid...
    # first make a temp 2d array of vertices
    w, h = 6, 7
    random_offset = lambda: -.3+random.random()*0.6
    vert_arr = [[Vertex((x+random_offset(),y+random_offset()))
                 for y in range(h)] for x in range(w)]
    # use temp lists to help with all the linking during DCEL construction
    down_edges = [[None for y in range(h-1)] for x in range(w)]
    right_edges = [[None for y in range(h)] for x in range(w-1)]
    faces = []
    # make edges to right and below each vert (making 4 half-edges each y loop-iteration)
    for x in range(w):
        for y in range(h):
            vert = vert_arr[x][y]
            vert.set_puzzle_node(PuzzleNode())
            if x < w-1:
                right_v = vert_arr[x+1][y]
                right_edge = HalfEdge(vert)
                right_twin = HalfEdge(right_v)
                right_edge.set_twin(right_twin)
                right_edges[x][y] = right_edge
            if y < h-1:
                down_v = vert_arr[x][y+1]
                down_edge = HalfEdge(vert)
                down_twin = HalfEdge(down_v)
                down_edge.set_twin(down_twin)
                down_edges[x][y] = down_edge
            if x < w - 1 and y < h - 1:
                face = Face(down_edge)
                if(random.randint(0,10) == 0):
                    t_col = Colors.BLACK if random.randint(0,1) == 0 else Colors.WHITE
                    face.set_token(TokenFactory.make_color_block_token(t_col))
                    
                face.color = (x*(200//w),40,y*(230//h))
                faces.append(face)
            if x > 0 and y > 0:
                # connection time
                # top edge - next
                top = right_edges[x-1][y-1].twin
                down = down_edges[x-1][y-1]
                top.next = down
                down.prev = top
                # down edge - next
                right = right_edges[x-1][y]
                down.next = right
                right.prev = down
                # right edge - next
                up = down_edges[x][y-1].twin
                right.next = up
                up.prev = right
                # up edge - next
                up.next = top
                top.prev = up
                
                #print([top, down, right, up])
    # flatten 2d array
    vert_arr = sum(vert_arr, [])
    for v in vert_arr:
        v.set_puzzle_node(PuzzleNode())
        if random.randint(0,w) == 0:
            v.puzzle_node.is_start = True
    # merge edges
    edges = sum(down_edges,[]) + sum(right_edges, [])
    graph = PuzzleGraph(vert_arr, edges, faces)
    return graph

# base class for different kinds of puzzles
# old: I made this when I thought it was still going to be purely a grid
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
                # place cellse
                
                # is it solvable?
                if solvable:
                    satisfied = True
                try_count -= 1
                
def main():
    pygame.init()
    screen = pygame.display.set_mode((600,600))
    screen.fill((250,250,250))
    running = True
    test_graph = make_test_graph()
    puzzle_surf = pygame.Surface((500,500))
    puzzle_surf.fill((155,155,155))
    test_graph.draw_to_fit(puzzle_surf)
    screen.blit(puzzle_surf, (50,50))
    pygame.display.update()
    while running:
        for e in pygame.event.get():
            if e.type == QUIT:
                running = False
            elif e.type == KEYDOWN:
                if e.key == K_ESCAPE:
                    running = False
        #screen.fill((0,0,0))    
        #pygame.display.update()
    pygame.quit()
if __name__ == "__main__":
    main()
