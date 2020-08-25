# trapezoid.py

def TrapezoidalProfile(t1,t2,tf,v):
    
    # Create a numerical array and fill with time values (1st column) 
    # and velocity values (2nd column)
    t = [0., t1, t2, tf]
    v = [0., v, v, 0.]

    return t, v
