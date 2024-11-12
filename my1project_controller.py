import robot

def main():
    red = 0
    green = 0
    blue = 0
    range = 0.0
    red_seen = False  
    green_seen = False  
    blue_seen = False 
    
    robot1 = robot.ARAP()
    robot1.init_devices()
    
    while True:
        robot1.reset_actuator_values()
        range = robot1.get_sensor_input()  
        robot1.blink_leds()
        red, green, blue = robot1.get_camera_image(5)  

       
        if red > green and red > blue:  
            if not red_seen:
                print("I see red")
                red_seen = True
            summary(red_seen, green_seen, blue_seen)
        
        elif green > red and green > blue:  
            if not green_seen:
                print("I see green")
                green_seen = True
            summary(red_seen, green_seen, blue_seen)
        
        elif blue > red and blue > green:  
            if not blue_seen:
                print("I see blue")
                blue_seen = True
            summary(red_seen, green_seen, blue_seen)

        if robot1.front_obstacles_detected(): 
            robot1.move_backward()  
            robot1.turn_left() 
        else:
            robot1.run_braitenberg()  

        robot1.set_actuators()  
        robot1.step()  

# 
def summary(red_seen, green_seen, blue_seen):
    encountered = []
    if red_seen:
        encountered.append("Red")
    if green_seen:
        encountered.append("Green")
    if blue_seen:
        encountered.append("Blue")
    
    print(f"Encountered so far: {', '.join(encountered) if encountered else 'None'}")

if __name__ == "__main__":
    main()
