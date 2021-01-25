import numpy as np 

def get_SP(SP_type, SP_step=0.02, SP_time=13.0, SP_val=8.0):
    
    if(SP_type=='constant'):
        
        t_num = int(SP_time/SP_step) 
        t = np.linspace(0.0, SP_time, t_num)
        SP = SP_val*np.ones(t_num)
        return np.deg2rad(SP), t
    
    elif(SP_type=='four_step'):
        
        A1 = np.random.randint(-10,15)
        A2 = np.random.randint(-10,15)
        A3 = np.random.randint(-10,15)
        A4 = np.random.randint(-10,15)
                
        t_num = int(SP_time/SP_step) 
        t = np.linspace(0.0, SP_time, t_num)
        delta_t = SP_time/4.0
        
        SP = A1*np.ones(t_num)
        
        SP[int(1*delta_t/SP_step):int(2*delta_t/SP_step)]=A2
        SP[int(2*delta_t/SP_step):int(3*delta_t/SP_step)]=A3
        SP[int(3*delta_t/SP_step):int(4*delta_t/SP_step)]=A4
        
        return np.deg2rad(SP), t
    
    elif(SP_type=='complex_random'):
      
        A1 = np.random.randint(-10,15)
        A2 = np.random.randint(-10,15)
        A3 = np.random.randint(-10,15)
        A4 = np.random.randint(-10,15)
        A5 = np.random.randint(-10,0)
        A6 = np.random.randint(0,10)
        A7 = np.random.randint(-10,0)
        A8 = np.random.randint(0,10)
        A9 = np.random.randint(2,6)
        T9 = np.random.randint(5,25)
        
        SP_time = 26.0
        t_num = int(SP_time/SP_step) 
        t = np.linspace(0.0, SP_time, t_num)
        delta_t = SP_time/13.0
        
        SP = (A1)*np.ones(t_num)
        
        SP[int(1*delta_t/SP_step):int(2*delta_t/SP_step)]=A2
        SP[int(2*delta_t/SP_step):int(3*delta_t/SP_step)]=A3
        SP[int(3*delta_t/SP_step):int(4*delta_t/SP_step)]=A4
        ramp1 = np.ones(100)
        ramp2 = np.ones(100)
        ramp3 = np.ones(100)
        ramp4 = np.ones(100)
        for i in range(100):
          ramp1[i] = (A4 + i*A5/(99))
          ramp2[i] = (A5+A4 + i*A6/(99))
          ramp3[i] = (A6+A5+A4 + i*A7/(99))
          ramp4[i] = (A7+A6+A5+A4 + i*A8/(99))    

        SP[int(4*delta_t/SP_step):int(5*delta_t/SP_step)]= ramp1
        SP[int(5*delta_t/SP_step):int(6*delta_t/SP_step)]= ramp2
        SP[int(6*delta_t/SP_step):int(7*delta_t/SP_step)]= ramp3
        SP[int(7*delta_t/SP_step):int(8*delta_t/SP_step)]= ramp4
        SP[int(8*delta_t/SP_step):int(9*delta_t/SP_step)]= ramp4[-1]

        SP[int(9*delta_t/SP_step):int(13*delta_t/SP_step)] = (ramp4[-1]*np.ones(400) + A9 * np.sin(np.linspace(0.0, T9, 400)))

        return np.deg2rad(SP), t
  
    elif(SP_type=='complex_purposive'):

        A1 = 8.0
        A2 = -10.0
        A3 = 5.0
        A4 = 0.0
        A5 = 15.0
        A6 = -20
        A7 = 9
        A8 = -4
        A9 = 6
        T9 = 18
        
        SP_time = 13.0
        t_num = int(SP_time/SP_step) 
        t = np.linspace(0.0, SP_time, t_num)
        delta_t = SP_time/13.0
      
        SP = (A1)*np.ones(t_num)
      
        SP[int(1*delta_t/SP_step):int(2*delta_t/SP_step)]=A2
        SP[int(2*delta_t/SP_step):int(3*delta_t/SP_step)]=A3
        SP[int(3*delta_t/SP_step):int(4*delta_t/SP_step)]=A4
        ramp1 = np.ones(50)
        ramp2 = np.ones(50)
        ramp3 = np.ones(50)
        ramp4 = np.ones(50)
        for i in range(50):
          ramp1[i] = (A4 + i*A5/(49))
          ramp2[i] = (A5+A4 + i*A6/(49))
          ramp3[i] = (A6+A5+A4 + i*A7/(49))
          ramp4[i] = (A7+A6+A5+A4 + i*A8/(49))    

        SP[int(4*delta_t/SP_step):int(5*delta_t/SP_step)]= ramp1
        SP[int(5*delta_t/SP_step):int(6*delta_t/SP_step)]= ramp2
        SP[int(6*delta_t/SP_step):int(7*delta_t/SP_step)]= ramp3
        SP[int(7*delta_t/SP_step):int(8*delta_t/SP_step)]= ramp4
        SP[int(8*delta_t/SP_step):int(9*delta_t/SP_step)]= ramp4[-1]

        SP[int(9*delta_t/SP_step):int(13*delta_t/SP_step)] = (ramp4[-1]*np.ones(200) + A9 * np.sin(np.linspace(0.0, T9, 200)))
        SP[-1] = SP[-2]

        return np.deg2rad(SP), t

    elif(SP_type=='complex_purposive_v2'):
      
        A1 = 10.0
        A2 = 5.0
        A3 = -15.0
        A4 = 12.0
        A5 = -8.0
        A6 = 8.0
        A7 = -2.0
        A9 = 8.0
        A10 = 6.0
        A11 = 4.0
        T8 =1.0
        T9 = 2.0
        T10 = 3.0
        
        SP_time = 13.0
        t_num = int(SP_time/SP_step) 
        t = np.linspace(0.0, SP_time, t_num)
        delta_t = SP_time/13.0
        
        SP = (A1)*np.ones(t_num)
        SP[int(1*delta_t/SP_step):int(2*delta_t/SP_step)]=A2

        ramp1 = np.ones(50)
        ramp2 = np.ones(50)
        ramp3 = np.ones(50)
        ramp4 = np.ones(50)
        ramp5 = np.ones(50)

        for i in range(50):

          ramp1[i] = (A2 + i*A3/(49))
          ramp2[i] = (A3+A2 + i*A4/(49))
          ramp3[i] = (A4+A3+A2 + i*A5/(49))
          ramp4[i] = (A5+A4+A3+A2 + i*A6/(49))
          ramp5[i] = (A6+A5+A4+A3+A2 + i*A7/(49))

        SP[int(2*delta_t/SP_step):int(3*delta_t/SP_step)] = ramp1
        SP[int(3*delta_t/SP_step):int(4*delta_t/SP_step)] = ramp2
        SP[int(4*delta_t/SP_step):int(5*delta_t/SP_step)]= ramp3
        SP[int(5*delta_t/SP_step):int(6*delta_t/SP_step)]= ramp4
        SP[int(6*delta_t/SP_step):int(7*delta_t/SP_step)]= ramp5
        SP[int(7*delta_t/SP_step):int(8*delta_t/SP_step)]= ramp5[-1]

        SP[int(8*delta_t/SP_step):int(9*delta_t/SP_step)] = (ramp5[-1]*np.ones(50) + A9 * np.sin(np.linspace(0.0, 2*3.14*T8, 50)))
        SP[int(9*delta_t/SP_step):int(11*delta_t/SP_step)] = (ramp5[-1]*np.ones(100) + A10 * np.sin(np.linspace(0.0, 2*3.14*T9, 100)))
        SP[int(11*delta_t/SP_step):int(13*delta_t/SP_step)] = (ramp5[-1]*np.ones(100) + A11 * np.sin(np.linspace(0.0, 2*3.14*T10, 100)))
        SP[-1] = SP[-2]

        return np.deg2rad(SP), t



  