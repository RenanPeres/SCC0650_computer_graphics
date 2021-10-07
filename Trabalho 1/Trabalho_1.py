# Trabaho Prático 1
#
# André Baconcelo Prado Furlanetti - N°USP
# Renan Peres Martins - N°USP: 10716612

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import math

TAM = 0.1
TAM3 = 0.05
t_x =  0.0
t_y = 0.0
p_x =  0.0 
p_y = 0.0
s =  1.0
angle = 0.0
angle2 = 0.0


def multiplica_matriz(a,b):
    m_a = a.reshape(4,4)
    m_b = b.reshape(4,4)
    m_c = np.dot(m_a,m_b)
    c = m_c.reshape(1,16)
    return c

def key_event(window,key,scancode,action,mods):
    global p_x, p_y, s, angle2
    # Primeira figura
    if (key == 68 and p_x <  0.85): p_x += 0.01 # tecla D
    if(key == 65 and p_x > -0.85): p_x -= 0.01 # tecla A

    if(key == 87 and p_y <  0.62): p_y += 0.01 # tecla W
    if(key == 83 and p_y > -0.75): p_y -= 0.01 # tecla S

    if(key == 265 and s < 2.0): s += 0.01 # seta para cima
    if(key == 264 and s > 0.5): s -= 0.01 # seta para baixo   

    # Segunda figura
    if(key == 262): angle2 += 0.01 # seta da direita
    if(key == 263): angle2 -= 0.01 # seta da esquerda


# Entrada: angulo de longitude, latitude, raio
# Saida: coordenadas na esfera
def F(u,v,r):
    x = r*math.sin(v)*math.cos(u)
    y = r*math.sin(v)*math.sin(u)
    z = r*math.cos(v)
    return (x,y,z)

def drawFirstObject(vertices):
    vertices['position'][0] = [0.0, 0.0]

    step = np.pi/8
    ang = 0.0

    for i in range(17):
        if(i % 2 == 1):
            vertices['position'][i] = [0.6 * TAM * np.cos(ang),0.6 * TAM * np.sin(ang)]
        else:
            vertices['position'][i] = [0.3 * TAM * np.cos(ang), 0.3 * TAM * np.sin(ang)]

        ang += step
    
    vertices['position'][17] = vertices['position'][1]

def drawSecondObject(vertices):
    step = np.pi/6
    ang = 0.0

    for i in range(18,66,4):
        smallstep = step/7

        vertices['position'][i] = [0.0,0.0]

        vertices['position'][i+1] = [0.7 * TAM * np.cos(ang + 2 * smallstep),0.7 * TAM * np.sin(ang + 2 * smallstep)]

        vertices['position'][i+2] = [1.0 * TAM * np.cos(ang + 4 * smallstep),1.0 * TAM * np.sin(ang + 4 * smallstep)]

        vertices['position'][i+3] = [0.7 * TAM * np.cos(ang + 6 * smallstep),0.7 * TAM * np.sin(ang + 6 * smallstep)]    

        ang += step

    vertices['position'][66] = [0.002,0.0]

    vertices['position'][67] = [-0.002,0.0]

    vertices['position'][68] = [-0.002,-0.3]
    
    vertices['position'][69] = [+0.002,-0.3]

def drawThirdObject(vertices):
    third = np.zeros(9, [("position", np.float32, 2)])
    third["position"] = [
        ( 0.0, 0.0),
        ( 1.0, 3.0),
        ( 0.0, 1.0),
        (-1.0, 3.0),
        ( 2.0, 0.0),
        ( 1.0, 1.0),
        (-1.0, 1.0),
        (-2.0, 0.0),
        ( 2.0, 0.0)
    ]
    
    for i in range(len(third)):
        vertices['position'][i + 70] = TAM3 * third["position"][i][0], TAM3 * third["position"][i][1]


def main():
    global angle, angle2
    glfw.init()
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(1000, 800, "Trabalho 1", None, None)
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

    # preparando espaço para 3 vértices usando 2 coordenadas (x,y)
    vertices = np.zeros(79, [("position", np.float32, 2)])

    drawFirstObject(vertices)   #Estrela
    drawSecondObject(vertices)  #Flor
    drawThirdObject(vertices)   #Passáro


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

    glVertexAttribPointer(loc, 2, GL_FLOAT, False, stride, offset)

    loc_color = glGetUniformLocation(program, "color")
    R = 1.0
    G = 0.0
    B = 0.0

    glfw.set_key_callback(window,key_event)

    glfw.show_window(window)

    x = 0.0
    step = np.pi/900
    mult = 1.0
    angle = 0.0

    while not glfw.window_should_close(window):

        glfw.poll_events() 

        
        glClear(GL_COLOR_BUFFER_BIT) 
        glClearColor(0.0, 0.0, 0.0, 1.0)

        loc = glGetUniformLocation(program, "mat_transformation");

        mat_aux = np.zeros(16)
        mat_final = np.zeros(16)

        #### PRIMEIRO OBJETO
        t_x = x
        t_y = mult * 0.1 * np.sin(4*x) # Faz o caminho da senoide
        
        if(np.fabs(t_y) < 0.001 and np.fabs(x) > 0.001):
            step *= (-1)
            mult *= (-1)
        
            
        x += step

        # MATRIZ DE ESCALA
        mat_scale = np.array([
            s   , 0.0, 0.0, 0.0,
            0.0,    s, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        ])
        
        # MATRIZ DE ROTAÇÃO
        mat_rotation = np.array([
            np.cos(angle), -np.sin(angle), 0.0, 0.0,
            np.sin(angle),  np.cos(angle), 0.0, 0.0,
            0.0      ,  0.0      , 1.0, 0.0,
            0.0      ,  0.0      , 0.0, 1.0
        ])

        # MATRIZ DE TRANSLAÇÃO
        mat_translation = np.array([
            1.0, 0.0, 0.0, t_x ,
            0.0, 1.0, 0.0, t_y + 0.5,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        ])

        # Multiplica as matrizes
        mat_aux = multiplica_matriz(mat_rotation, mat_scale)
        mat_final = multiplica_matriz(mat_translation, mat_aux)

        # Para deixar a estrela girando
        angle -= 0.005

        #Enviando a matriz de transformação para essa figura
        glUniformMatrix4fv(loc, 1, GL_TRUE, mat_final)

	    #Renderizando a cor
        glUniform4f(loc_color, 254.0/255.0, 254.0/255.0, 68.0/255.0, 1.0)
        glDrawArrays(GL_TRIANGLE_FAN, 0, 18)

        ####### SEGUNDO OBJETO
        
        # Alterando a matriz de translação para o segundo objeto
        mat_translation[3] = -0.6
        mat_translation[7] = -0.55

        glUniformMatrix4fv(loc, 1, GL_TRUE, mat_translation)

        # Cabo da flor
        glUniform4f(loc_color, 0.0, 128.0/255.0, 0.0/255.0, 1.0) # Cor verde
        glDrawArrays(GL_TRIANGLE_STRIP, 66, 4)

        # Alterando as matrizes de rotação e escala para o segundo objeto
        mat_scale[0] = 1.0
        mat_scale[5] = 1.0

        mat_rotation[0] =  np.cos(angle2)
        mat_rotation[1] = -np.sin(angle2)
        mat_rotation[4] =  np.sin(angle2)
        mat_rotation[5] =  np.cos(angle2)        
        
        mat_aux = multiplica_matriz(mat_rotation, mat_scale)
        mat_final = multiplica_matriz(mat_translation, mat_aux)

        # enviando a matriz de transformacao para a GPU, vertex shader, variavel mat_transformation
        glUniformMatrix4fv(loc, 1, GL_TRUE, mat_final)
    
        # Cor roxa das pétalas
        glUniform4f(loc_color, 150.0/255.0, 0.0, 205.0/255.0, 1.0)
        
        for j in range(0,12):
            glDrawArrays(GL_TRIANGLE_FAN, 18 + 4*j, 4)

        #### TERCEIRO OBJETO

        # Alterando a matriz de translação para o terceiro objeto
        mat_translation[3] = p_x
        mat_translation[7] = p_y        

        glUniformMatrix4fv(loc, 1, GL_TRUE, mat_translation)

	    #renderizando
        glUniform4f(loc_color, 107.0/255.0, 142.0/255.0, 35.0/255.0, 1.0)
        glDrawArrays(GL_TRIANGLE_FAN, 70, 4)
        glDrawArrays(GL_TRIANGLE_STRIP, 74, 5)

        glfw.swap_buffers(window)

    glfw.terminate()


if __name__ == "__main__":
  main()