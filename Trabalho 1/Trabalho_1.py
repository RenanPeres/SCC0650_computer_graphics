# Trabaho Prático 1
#
# André Baconcelo Prado Furlanetti - N°USP
# Renan Peres Martins - N°USP: 10716612

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np

glfw.init()
glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
window = glfw.create_window(700, 700, "Exercício aula 4", None, None)
glfw.make_context_current(window)

vertex_code = """
        attribute vec3 position;
        uniform mat4 mat_transformation;
        void main(){
            gl_Position = mat_transformation * vec4(position,1.0);
        }
        """
fragment_code = """
        uniform vec4 color;
        void main(){
            gl_FragColor = color;
        }
        """

# Request a program and shader slots from GPU
program  = glCreateProgram()
vertex   = glCreateShader(GL_VERTEX_SHADER)
fragment = glCreateShader(GL_FRAGMENT_SHADER)

# Set shaders source
glShaderSource(vertex, vertex_code)
glShaderSource(fragment, fragment_code)

# Compile shaders
glCompileShader(vertex)
if not glGetShaderiv(vertex, GL_COMPILE_STATUS):
    error = glGetShaderInfoLog(vertex).decode()
    print(error)
    raise RuntimeError("Erro de compilacao do Vertex Shader")

glCompileShader(fragment)
if not glGetShaderiv(fragment, GL_COMPILE_STATUS):
    error = glGetShaderInfoLog(fragment).decode()
    print(error)
    raise RuntimeError("Erro de compilacao do Fragment Shader")

# Attach shader objects to the program
glAttachShader(program, vertex)
glAttachShader(program, fragment)

# Build program
glLinkProgram(program)
if not glGetProgramiv(program, GL_LINK_STATUS):
    print(glGetProgramInfoLog(program))
    raise RuntimeError('Linking error')
    
# Make program the default program
glUseProgram(program)

vertices = np.zeros(1728, [("position", np.float32, 3)])

#Declaração de vértices
#
#
#

# Request a buffer slot from GPU
buffer = glGenBuffers(1)
# Make this buffer the default one
glBindBuffer(GL_ARRAY_BUFFER, buffer)

# Upload data
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)
glBindBuffer(GL_ARRAY_BUFFER, buffer)

# Bind the position attribute
# --------------------------------------
stride = vertices.strides[0]
offset = ctypes.c_void_p(0)

loc = glGetAttribLocation(program, "position")
glEnableVertexAttribArray(loc)

glVertexAttribPointer(loc, 3, GL_FLOAT, False, stride, offset)

loc_color = glGetUniformLocation(program, "color")

# exemplo para matriz de translacao
# t_x = 0
# t_y = 0

# def key_event(window,key,scancode,action,mods):
#     global t_x, t_y
    
# #     print('[key event] key=',key)
# #     print('[key event] scancode=',scancode)
# #     print('[key event] action=',action)
# #     print('[key event] mods=',mods)
# #     print('-------')
#     if key == 265: t_y += 0.01 #cima
#     if key == 264: t_y -= 0.01 #baixo
#     if key == 263: t_x -= 0.01 #esquerda
#     if key == 262: t_x += 0.01 #direita
    
# glfw.set_key_callback(window,key_event)

glfw.show_window(window)


import math
d = 0.0
glEnable(GL_DEPTH_TEST) ### importante para 3D

def multiplica_matriz(a,b):
    m_a = a.reshape(4,4)
    m_b = b.reshape(4,4)
    m_c = np.dot(m_a,m_b)
    c = m_c.reshape(1,16)
    return c

t_y = -0.25
t_x =  0.0
s_x = -0.5
s_y = -0.5
s_z = -0.5

while not glfw.window_should_close(window):

    glfw.poll_events() 

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glClearColor(1.0, 1.0, 1.0, 1.0)
    

    mat_translation = np.array([    1.0, 0.0, 0.0, t_x, 
                                    0.0, 1.0, 0.0, t_y, 
                                    0.0, 0.0, 1.0, 0.0, 
                                    0.0, 0.0, 0.0, 1.0], np.float32)

    mat_scale       = np.array([    s_x, 0.0, 0.0, 0.0, 
                                    0.0, s_y, 0.0, 0.0, 
                                    0.0, 0.0, s_z, 0.0, 
                                    0.0, 0.0, 0.0, 1.0], np.float32)

    mat_transform = multiplica_matriz(mat_translation,mat_scale)

    ### apenas para visualizarmos a esfera rotacionando
    d -= 0.002 # modifica o angulo de rotacao em cada iteracao
    cos_d = math.cos(d)
    sin_d = math.sin(d)

    mat_rotation_z = np.array([     cos_d, -sin_d, 0.0, 0.0, 
                                    sin_d,  cos_d, 0.0, 0.0, 
                                    0.0,      0.0, 1.0, 0.0, 
                                    0.0,      0.0, 0.0, 1.0], np.float32)
    
    mat_rotation_x = np.array([     1.0,   0.0,    0.0, 0.0, 
                                    0.0, cos_d, -sin_d, 0.0, 
                                    0.0, sin_d,  cos_d, 0.0, 
                                    0.0,   0.0,    0.0, 1.0], np.float32)
    
    mat_rotation_y = np.array([     cos_d,  0.0, sin_d, 0.0, 
                                    0.0,    1.0,   0.0, 0.0, 
                                    -sin_d, 0.0, cos_d, 0.0, 
                                    0.0,    0.0,   0.0, 1.0], np.float32)
    
    mat_transform = multiplica_matriz(mat_rotation_y,mat_transform)

    loc = glGetUniformLocation(program, "mat_transformation")
    glUniformMatrix4fv(loc, 1, GL_TRUE, mat_transform)

    glPolygonMode(GL_FRONT_AND_BACK,GL_LINE)
    
    for triangle in range(0,len(vertices),3):
        
        glDrawArrays(GL_TRIANGLES, triangle, 3)     

    glfw.swap_buffers(window)

glfw.terminate()