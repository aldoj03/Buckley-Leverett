from tkinter import Toplevel, font, ttk
from tkinter import LabelFrame, Entry, Label, LEFT, W, E, Tk
from matplotlib import collections
import numpy as np
from scipy import optimize 
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
from pylab import figure,   ylabel

class App:

    def __init__(self,window):
        self.win = window
        self.win.title('Buckley Leverett')
        self.createLayout()

  

    def createLayout(self):
        frame = LabelFrame(self.win, text = 'Insetar valores',font=('',19))
        frame.grid( row = 0, column = 0, columnspan = 3, pady= 20, padx=20)
        
        Label(frame,font = ('',12),justify=LEFT, text= 'Saturación de agua inicial: ').grid(row=1, column= 0,pady=5)
        self.initial_sw = Entry(frame, font=('',12))
        self.initial_sw.grid(row=1 , column= 1)
        self.initial_sw.insert(0,0.2)
        self.initial_sw.focus()

        Label(frame,font = ('',12), justify=LEFT,text= 'Saturación de agua residual: ').grid(row=2, column= 0,pady=5, padx= 5)
        self.residual_w = Entry(frame, font=('',12))
        self.residual_w.insert(0,0.2)
        self.residual_w.grid(row=2 , column= 1,padx= 5)

        Label(frame,font = ('',12), justify=LEFT, text= 'Saturación de aceite residual: ').grid(row=3, column= 0, pady=5, padx= 5)
        self.residual_o = Entry(frame, font=('',12))
        self.residual_o.insert(0,0.2)
        self.residual_o.grid(row=3 , column= 1,padx= 5)


        Label(frame,font = ('',12), justify=LEFT,text= 'Viscosidad del agua: ').grid(row=1, column= 4, pady=5,padx= 5 )
        self.viscosity_w = Entry(frame, font=('',12))
        self.viscosity_w.insert(0,1)
        self.viscosity_w.grid(row=1 , column= 5, padx= 5)


        Label(frame,font = ('',12), justify=LEFT,text= 'Viscosidad del aceite: ').grid(row=2, column= 4,pady=5, padx= 5)
        self.viscosity_o = Entry(frame, font=('',12))
        self.viscosity_o.insert(0,2)
        self.viscosity_o.grid(row=2 , column= 5,padx= 5)
        
        
        Label(frame,font = ('',12), justify=LEFT,text= 'Porosidad: ').grid(row=3, column= 4,pady=5, padx= 5)
        self.poro = Entry(frame, font=('',12))
        self.poro.insert(0,0.25)
        self.poro.grid(row=3 , column= 5,padx= 5)

        Label(frame,font = ('',12), justify=LEFT,text= 'Inyección de agua: ').grid(row=4, column= 0,pady=5, padx= 5)
        self.qi = Entry(frame, font=('',12))
        self.qi.insert(0,900)
        self.qi.grid(row=4 , column= 1,padx= 5)

        Label(frame,font = ('',12), justify=LEFT,text= 'Área de sección transversal (pie²): ').grid(row=4, column= 4,pady=5, padx= 5)
        self.xarea = Entry(frame, font=('',12))
        self.xarea.insert(0,26400)
        self.xarea.grid(row=4 , column= 5,padx= 5)


  


        frame_tablas = LabelFrame(frame, text='Tabla de valores', font=('',15))
        frame_tablas.grid( row = 6, column = 0, pady= 20, padx=20)
        
        frame_dias = LabelFrame(frame, text='Días a evaluar', font=('',15))
        frame_dias.grid( row = 6, column = 1, pady= 20, padx=20)
        
        self.dataTables(frame_tablas, 7)
        self.diasInputs(frame_dias)
        ttk.Button(frame, text='Generar gráficas',  command=self.sendData).grid(row= 6, columnspan=2, column=3, pady= 20, sticky = W + E)

    def sendData(self):
        res = self.validateNumbers()
        # if (res):
        buc =  BuckleyLev()
        init_vals(self.korkrw,self.sw,self.xarea,self.qi,self.poro,self.viscosity_o,self.viscosity_w,self.residual_o,self.residual_w,self.initial_sw, self.intervalos)
        buc.run_graphs()
        # else:
            # tkinter.messagebox.showwarning(title='Error', message='Campos requeridos')


    def dataTables(self,frame,iniRow):
        self.sw = []
        self.korkrw = []
        krejpm = [30.23, 17.00, 9.56, 5.38, 3.02, 1.70, 0.96, 0.54, 0.30,0.17, 0.10]
        sw_data = [.25, .3 ,.35,.4,.45,.5,.55,.6,.65,.7,.75]
        Label(frame,font = ('',12), justify=LEFT,text= 'Sw').grid(row=iniRow, column= 0,pady=5, padx= 5)
       

        Label(frame,font = ('',12), justify=LEFT,text= 'Kro/Krw').grid(row=iniRow, column= 1,pady=5, padx= 5)
       
        for i in range(11):
            sw = Entry(frame, font=('',12))
            sw.grid(row=iniRow + i + 1 , column= 0,padx= 5, pady=5)
            sw.insert(0,sw_data[i])
            self.sw.append(sw)

            korkrw = Entry(frame, font=('',12))
            korkrw.insert(0,krejpm[i])
            korkrw.grid(row=iniRow + i + 1 , column= 1,padx= 5, pady=5)
            self.korkrw.append(korkrw)

    def diasInputs(self,frame):
        diasIniciales = [60, 120, 240]
        self.intervalos = []
        for i in range(3):
            number = i +1 
            row = 6 + i
            Label(frame,font = ('',12), justify=LEFT,text= 'Intervalo ' + str(number) ).grid(row=row, column= 0,pady=5, padx= 5)
            dia = Entry(frame, font=('',12))
            dia.insert(0,diasIniciales[i])
            dia.grid(row=row , column= 1,padx= 5)
            self.intervalos.append(dia)

       
    def validateNumbers(self):
        res = True
        count = len(self.sw)

        for i in range(count):
            if not(self.sw[i].get()):
                res = False


        return res
    
    def results_page(self, a):
        self.new_page = Toplevel(self.win)
        self.new_page.minsize(width=200, height=300)
        lista = ttk.Treeview(self.new_page)
        titles = ['Sw','Kro/Krw','Fw','dfw/dSw']
        for i in range(3):
            titulo = 'xf(' + str(round(a.dias[i])) +')'
            titles.append(titulo)
        

        lista['columns']=titles
        for i in range(3):
            lista.column(titles[4 + i],width=100)
            lista.heading(titles[4 + i], text = titles[4 + i])
        lista.column('#0', width=0)
        lista.column('Sw',width=100)
        # lista.column('Krw',width=100)
        # lista.column('Kro',width=100)
        lista.column('Kro/Krw',width=100)
        lista.column('Fw',width=100)
        lista.column('dfw/dSw',width=100)

        lista.heading('#0', text = '')
        lista.heading('Sw', text = 'Sw')
        # lista.heading('Krw', text = 'Krw')
        # lista.heading('Kro', text = 'Kro')
        lista.heading('Kro/Krw', text = 'Kro/Krw')
        lista.heading('Fw', text = 'Fw')
        lista.heading('dfw/dSw', text = 'dfw/dSw')


        lista.grid(row=0, column= 0)
        timeTmp1 = sorted((a.slice_rarefaction(a.dias[0])[1]))
        time1 = []
        for index in range(11):
            if(index < len(timeTmp1)):
                time1.append(round(timeTmp1[index]))
            else:
                time1.append(round(timeTmp1[len(timeTmp1)-1]))

        print(time1)
        time1.reverse()

        timeTmp2 = sorted((a.slice_rarefaction(a.dias[1])[1]))
        time2 = []
        for index in range(11):
            if(index < len(timeTmp2)):
                time2.append(round(timeTmp2[index]))
            else:
                time2.append(round(timeTmp2[len(timeTmp2)-1]))

        time2.reverse()

        timeTmp3 = sorted((a.slice_rarefaction(a.dias[1])[1]))
        time3 = []
        for index in range(11):
            if(index < len(timeTmp3)):
                time3.append(round(timeTmp3[index]))
            else:
                time3.append(round(timeTmp3[len(timeTmp3)-1]))
                
        time3.reverse()

    


        for i in range(11):
            fw = round(a.fractional_flow(i), 3) 
            dfw = round(a.fractional_flow_deriv(i),3)
            t1 =  round(time1[i]) 
            t2 =  round(time2[i]) 
            t3 =  round(time3[i]) 
            lista.insert(parent='', index=i +1 , iid=i + 1, text='', values=(a.sw[i],a.korkrw[i],fw,dfw,t1,t2,t3))


        
class BuckleyLev():

    def __init__(self):
       
        self.params = {
        #non wetting phase viscosity
            "viscosity_o": 2,
            #wetting phase viscosity
            "viscosity_w": 1,
            #initial water sat
            "initial_sw":0.2,
            #residual water saturation, Swirr
            "residual_w":0.2,
            #residual oil saturation, Sor
            "residual_o":0.2,
            #connate water saturation 
            #water rel perm at water curve end point
            "krwe":1,
            #oil rel perm at oil curve end point
            "kroe": 0.9,
            #dimless velocity results
            'vd_array':[],
            #porosity
            'poro':0.25,
            #water injection rate units 
            "inject_rate":900,
            #cross sectional area units m2
            "x-area":26400
        }


def k_rw(self,sw):
    #water relative perm calculation for a given water saturation 
    p = 11.174
    return ((self.params['krwe']))*sw**p


BuckleyLev.k_rw = k_rw

def k_rn(self,sw):
    #oil relative perm calculation for a given water saturation 
    q = 3.326
    return ((1.-self.params['kroe']*sw)**q)

BuckleyLev.k_rn = k_rn

def fractional_flow(self,sw):
    #returns the fractional flow
    # return 1./(1.+((korkrw[sw].get())*(self.viscosity_w/self.viscosity_o)))

   return 1./(1.+self.korkrw[sw] * self.viscosity_w/self.viscosity_o)

BuckleyLev.fractional_flow = fractional_flow


def plot_fractional_flow(self,):
    #plot the sw vs fractional flow
    # x = np.linspace(self.params["residual_w"]+1e-3,(1-self.params["residual_o"]+1e-3),100)
    x = BuckleyLev.sw
    y = [self.fractional_flow(i) for i in range(11) ]

  
    # print(y)
    # plt.plot(x,y)

    
    # sw_at_front = self.sw_at_shock_front()
    # plt.plot([sw_at_front],[self.fractional_flow(sw_at_front)],'ro')
    
    plt.title('Flujo fraccional en función de la saturacion de agua')
    plt.xlabel('Sw')
    plt.ylabel('Flujo fraccional')
    plt.ylim([0,1.1])
    plt.xlim([0,1])
    
    #add limiting fractional flow lines
    plt.hlines(y[len(y)-1],0,1,linestyles='dashed',lw=2, colors='0.4')
    plt.annotate('fw max: %.4f' % y[len(y)-1],xy=(0.08,0.98))

    
    cords_suavizadas = suavizar(x,x[0],x[10],y) 

    plt.plot(cords_suavizadas[0], cords_suavizadas[1])

    x = np.array(x)
    y = np.array(self.korkrw)
    
   
    # ab = optimize.curve_fit(lambda t,a,b: a*np.exp(b*t),  x,  y)
    ab = optimize.curve_fit(lambda t,a,b: a*np.exp(b*t),  x,  y)
    
    BuckleyLev.a = ab[0][0]
    BuckleyLev.b = ab[0][1]
    plt.show()
    #print(y[len(y)-1])
    
BuckleyLev.plot_fractional_flow = plot_fractional_flow

def sw_at_shock_front(self): 
    #first find range to search in by looking where 2nd derivative is negative (where curve turns)
    
    sw_start = 1.-self.residual_o
    sw_end = self.residual_w
    
    
    for sw in np.arange(self.residual_w+1.e-9, 1.-self.residual_o,0.001):
        if (self.fractional_flow_2deriv(sw)< -1.e-2 and sw < sw_start):
            #i.e. we are below sw_start and the grad is negative - update upper limit
            sw_start = sw
        if (self.fractional_flow_2deriv(sw)< -1.e-2 and sw > sw_end):
            # i.e. we are above sw_end and the gradient is negative - update lower limit
            sw_end = sw
        
        
    sw_at_front = 0.
    current_min = 1000.
        
    #find where finite difference approximation (almost) equals the continuous value of the function
    for sw in np.arange(sw_start,sw_end, 0.0001):
        
        current_diff = abs(self.fractional_flow_deriv(sw)-self.fractional_flow(sw)/sw)
        #print( 'grad func',self.fractional_flow_deriv(sw),sw,self.fractional_flow(sw)/sw)
        #print('approx',self.fractional_flow(sw)/sw, sw)
        if current_diff < current_min:
            #print('sw at front',current_min, sw,current_diff)
            current_min = current_diff
            sw_at_front = sw
            
            current_min = current_diff
                    
    return sw_at_front
BuckleyLev.sw_at_shock_front = sw_at_shock_front


def fractional_flow_deriv(self,sw):
    #calculate derivative of fractional flow - dFw/dSw - Vsh
   
    f_deriv = BuckleyLev.b*(self.fractional_flow(sw)**2 - self.fractional_flow(sw))
    
    return f_deriv

BuckleyLev.fractional_flow_deriv = fractional_flow_deriv

def fractional_flow_2deriv(self,sw):
    f_2deriv = ((self.fractional_flow(sw))-2*(self.fractional_flow(sw))-self.fractional_flow(sw))/((0.01)**2)
    return f_2deriv

BuckleyLev.fractional_flow_2deriv = fractional_flow_2deriv

def plot_fractional_flow_deriv(self):
    #plot the derivative dFw/dSw - Vsh vs Sw
    # x = np.linspace(self.params["residual_w"]+1e-3,(1-self.params["residual_o"]),100)
    x = BuckleyLev.sw


    # y = [self.fractional_flow_deriv(i) for i in x ]
    y = [self.fractional_flow_deriv(i) for i in range(11)]
    # plt.plot(x,y)
    cords_suavizadas = suavizar(x,x[0],x[10],y)
    plt.plot(cords_suavizadas[0], cords_suavizadas[1])

    plt.title('Derivada del flujo fraccional')
    plt.ylabel('dfw/dSw')
    plt.xlabel('Sw')
    plt.show()
    # show()
    
BuckleyLev.plot_fractional_flow_deriv=plot_fractional_flow_deriv

def frac_deriv_combo(self):
    x = BuckleyLev.sw

    # is the fractional flow curve
    y = [self.fractional_flow(i) for i in range(11) ]
    
    #y2 is the derivative
    # y2 = [self.fractional_flow_deriv(i) for i in x ]
    y2 = [self.fractional_flow_deriv(i) for i in range(11)]
 
    fig1 = figure()
    
    ax1 = fig1.add_subplot(111)
    line1 = ax1.plot(x,y,lw=2)
    ylabel("Fractional flow")
    ax1.set_ylim([0,1.2])
  
    ax2 = fig1.add_subplot(111, sharex=ax1, frameon=False)
    line2 = ax2.plot(x,y2,'-r',lw=2)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ylabel("Derivative dfw/dsw")
    ax2.set_xlim([0,1])
    
    plt.title('Fractional flow as a function of water saturation')
    
    plt.xlabel('Sw')
    
    ax1.hlines(y[len(y)-1],0,1,linestyles='dashed',lw=2, colors='0.4')
    ax1.annotate('fw max: %.4f' % y[len(y)-1],xy=(0.85,1.02))
    
    
    ax1.spines['left'].set_color('blue')
    ax1.tick_params(axis='y', colors='blue')
    
    ax1.spines['right'].set_color('red')
    ax2.tick_params(axis='y', colors='red')
    

    # show()
    
BuckleyLev.frac_deriv_combo = frac_deriv_combo



def shock_sat(self):
    
    #create dx window of increasing size for calculating welge

    x =BuckleyLev.sw


    y = [self.fractional_flow(i) for i in range(11) ]

  
    maximum, sw_shock = 0,0
    grads = []
    
    for index,item in enumerate(y):
        
        grad = (item - (y[0]+0.00001))/(x[index]-(x[0]+0.001)) 
        
        
        if grad>maximum:
            # plt.plot([self.params["residual_w"]+1e-3,x[index]],[0,item],'o-') 
            sw_shock = x[index]
            maximum = grad
    
    # plt.show()
    # print(sw_shock)
    return(sw_shock)
 

BuckleyLev.shock_sat = shock_sat



#generate xt curve for a given time step, if no time step, timwe =1 
def xt(self,time=1):
    
    # flux = (self.qi)/self.xarea
    # td = (flux*time)/(self.xarea*self.poro)
    flux = (5.615*self.qi * time)/(self.xarea*self.poro)
    y = BuckleyLev.sw

    x = [(self.fractional_flow_deriv(i)*flux) for i in range(11)]
    return(x,y)

BuckleyLev.xt = xt

def plot_xd(self,times=[1]):
    
    colors = {0:'r',1:'b',2:'c',3:'g'}
    max_val = 0
    
    for index,time in enumerate(times):
        if (max(self.xt(time)[0]) > max_val):
            max_val = max(self.xt(time)[0])
            
        plt.plot(self.xt(time)[0],self.xt(time)[1],colors[index],lw=2)

    plt.hlines(self.shock_sat(),0,1000,linestyles='dashed',lw=2, colors='0.4')
    
    plt.ylabel('dfw / dsw')
    plt.xlabel('Xsw ft')

    plt.annotate('Sw shock: %.2f' % self.shock_sat(),xy=(max_val-1.3,self.shock_sat()+0.012))
    plt.show()
    
BuckleyLev.plot_xd = plot_xd

def flw_dervi_horizontal(self):
    y = BuckleyLev.sw

    x = [(self.fractional_flow_deriv(i)) for i in range(11)]
    y2 = [self.fractional_flow(i) for i in range(11) ]
    
            
    plt.plot(x,y)
    plt.plot(y,y2)

    plt.hlines(self.shock_sat(),0,5,linestyles='dashed',lw=2, colors='0.4')
    
    plt.ylabel('dfw / dsw')
    plt.xlabel('Sw')

    # plt.annotate('Sw shock: %.2f' % self.shock_sat(BuckleyLev.korkrw),xy=(max_val-1.3,self.shock_sat(BuckleyLev.korkrw)+0.012))
    plt.show()

    
    plt.title('Fractional flow as a function of water saturation')
    
    plt.xlabel('Sw')
    
  
    



BuckleyLev.flw_dervi_horizontal = flw_dervi_horizontal

def slice_rarefaction(self,time):
    
    #generate the xt data 
    xt, sw = self.xt(time)
    index = 0

    #find out where the shock saturation is
    shock_sat = self.shock_sat()
    
    #now trim xt and sw arrays 
    for index, item in enumerate(sw):
        if item >= shock_sat:
            lim = index
            break
            
    sw_trim = (sw[lim:])
    xt_trim = (xt[lim:])
  
    return(sw_trim,xt_trim)
        
BuckleyLev.slice_rarefaction = slice_rarefaction


def plot_sat_front(self,times=[1]):
    colors = {0:'r',1:'b',2:'c',3:'g'}
       
    for index, time in enumerate(times):
        
        y = sorted((self.slice_rarefaction(time)[0]))
        x = sorted((self.slice_rarefaction(time)[1]),reverse=True)
        plt.plot(x,y,colors[index],lw=2)

        if (x[len(x)-1]>0.001):
            # plt.plot([0,x[len(x)-1]],[1,1],'b',lw=2)
            # plt.hlines(1-self.params["residual_o"],0,x[len(x)-1],colors[index],lw=2)
            plt.plot(x[len(x)-1],y[len(y)-1],colors[index]+'o')


        plt.vlines(x[0],y[0],0,colors[index],lw=2)
        plt.hlines(self.initial_sw,x[0],x[0]+1,colors[index],lw=2)

        #highlight vshock point
        plt.plot(x[0],y[0],colors[index]+'o') 
        
        plt.title('Avance frontal')
        plt.ylabel('Sw')
        plt.xlabel('Xsw ft')

    plt.ylim([0,1])
    plt.xlim([0,x[0]+1])
    
    plt.show()
BuckleyLev.plot_sat_front = plot_sat_front

def init_vals(korkrw,sw,xarea,qi,poro,viscosity_o,viscosity_w,residual_o,residual_w,initial_sw, dias):
    
    BuckleyLev.korkrw =  [float(korkrw[i].get()) for i in range(11) ]  
    BuckleyLev.dias =  [float(dias[i].get()) for i in range(3) ]  
    BuckleyLev.sw = [float(sw[i].get()) for i in range(11) ]
    BuckleyLev.xarea = float(xarea.get())
    BuckleyLev.qi = float(qi.get())
    BuckleyLev.poro = float(poro.get())
    BuckleyLev.viscosity_o = float(viscosity_o.get())
    BuckleyLev.viscosity_w = float(viscosity_w.get())
    BuckleyLev.residual_o = float(residual_o.get())
    BuckleyLev.residual_w = float(residual_w.get())
    BuckleyLev.initial_sw = float(initial_sw.get())


def suavizar(x, xini,xlast,y):
    xnew = np.linspace(xini, xlast , 300) 
    xshorted =  [np.nan_to_num(x[i]) for i in range(11) ]
    spl = make_interp_spline(xshorted, y, k=3)  # type: BSpline
    ynew = spl(xnew)
    return [xnew, ynew]

def run_graphs(self):
    a = BuckleyLev()
    a.plot_fractional_flow()
    a.plot_fractional_flow_deriv()
    # a.flw_dervi_horizontal()
    # a.frac_deriv_combo(korkrw)
    a.plot_xd(BuckleyLev.dias)
    # a.slice_rarefaction(1000)


    a.plot_sat_front(BuckleyLev.dias)
    application.results_page(a)
    # plt.show()
BuckleyLev.run_graphs = run_graphs


colors = {1:'r',2:'b'}
colors[1]+'o'

if __name__ == '__main__':
    window = Tk()
    application = App(window)
    window.mainloop()