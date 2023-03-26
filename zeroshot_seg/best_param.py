
best_prompt = {
    'airplane': [' The nose is the front part of the aircraft that houses the cockpit.', 'There is no wing part of an airplane in this grayscale map.', ' The back end of an airplane, with its two engines and two methods of steering.', 'This sentence is describing a partial airplane that is being shown in a depth map.'],
    'bag': ['The bag has a black strap that goes over the shoulder.', ' Ready for Literally AnythingThis bag is perfect for carrying all of your essentials with you on the go.'],
    'cap': ['Assuming you are talking about a baseball cap, the crown is typically the highest point of the hat, and the panels are the pieces of fabric that make up the sides and back of the hat.', 'This sentence is about the peak value of the grayscale depth map.'],
    'car': ['the car has a metal roof that is slanted down towards the back.', 'The hood of a car is an important part of the vehicle.', 'I need new tires for my car.', 'The engine is the heart of a car.'],
    'chair': ['Back of the chair depth map.', ' A closeup of the seat pad of a chair.', 'A leg part of a chair 3D model can look like a cylinder, a square, or a rectangle.', 'In a chair depth map, an armrest would appear as a horizontal line at the appropriate depth.'],
    'earphone': ['This is the earcup of a earphone in a three-dimensional map.', 'This is the headband part of an earphone in a 3D map.', 'A typical earphone has a wire that consists of four parts: the inner conductor, the dielectric insulation, the outer conductor, and the jacket.'],
    'guitar': ['This sentence is saying that the head or tuning pegs are the only part of the guitar that is shown in the depth map.', 'It is a representation of a gray 3D guitar model.', 'The body part of a guitar can be identified in this grayscale map by its shape.'],
    'knife': ['A depth map of a knife typically shows the blade as a thin, straight line, while the handle may be thicker and more curved.', 'A handle part of a knife 3D model might look like a cylindrical piece with a hole in the center for the blade to fit into.'],
    'lamp': ['There is no one-size-fits-all answer to this question, as the best way to segment the leg or wire part of a lamp in a depth map may vary depending on the.', 'Since this is a depth map, you can segment the lampshade by finding the points in the depth map that correspond to the lampshade.'],
    'laptop': ['The keyboard feature of a laptop 3D model is that it is a separate object that can be moved around and positioned as desired.', 'Laptop computer with screen open, viewed from above.'],
    'motorbike': ["The gas tank's motorbike would appear as a dark object in a grayscale depth map.", 'There is no easy answer for this question.', 'This sentence is describing a wheel on a motorcycle in a photograph.', 'There is no definitive answer to this question as it depends on the specific depth map and the desired outcome.', 'There is no definitive answer to this question since it will vary depending on the desired outcome.', 'The engine is the "heart" of the bike.'],
    'mug': ['This sentence is describing a depth map, which is a tool used in computer vision to create a representation of the surfaces of a scene from a set of digital images.', 'Only the bottom part of this mug is recognized.'],
    'pistol': ['This is the part of the pistol depth map that shows the barrel.', 'The part of the pistol that you would hold in your hand is the grip.', 'Thesynonym of this sentence is: The trigger and guard of a gun.'],
    'rocket': ['ROCKET BODYThis is the body of a rocket.', 'A fin is typically a thin, flat surface that is attached to the back end of a rocket.', 'A nose cone on a rocket 3D model typically looks like a cone or pyramid shape.'],
    'skateboard': ['The depth map of the wheel on a skateboard is important.', 'Caption: The deck of a skateboard, viewed from the top.', 'This sentence is describing a strap or belt that goes around the foot of a skateboard.'],
    'table': ['A depth map of a table, showing the desktop at the top and the underside of the table at the bottom.', 'The table is a rectangle with a light gray color.', 'The table is a rectangle with a light gray color.'],
}

best_vweight = {
    'airplane': [0.75, 0.75, 0.25, 0.25, 0.25, 0.50, 1.00, 0.25, 0.25, 0.25],
    'bag': [0.75, 0.75, 0.25, 0.75, 1.00, 0.25, 1.00, 0.50, 0.25, 0.25],
    'cap': [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    'car': [0.75, 0.75, 0.25, 0.25, 0.25, 0.75, 0.25, 0.75, 1.00, 0.25],
    'chair': [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    'earphone': [0.75, 0.75, 0.25, 0.25, 0.25, 0.25, 0.75, 0.50, 0.25, 0.50],
    'guitar': [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    'knife': [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    'lamp': [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    'laptop': [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    'motorbike': [0.75, 0.75, 0.25, 0.25, 0.50, 0.75, 0.25, 0.75, 1.00, 0.25],
    'mug': [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    'pistol': [0.75, 0.75, 0.25, 0.25, 0.25, 1.00, 0.25, 1.00, 0.75, 0.25],
    'rocket': [0.75, 0.75, 0.25, 0.25, 0.50, 1.00, 0.25, 0.50, 0.25, 0.75],
    'skateboard': [0.75, 0.75, 0.25, 0.50, 0.25, 1.00, 0.50, 0.25, 0.75, 1.00],
    'table': [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
}

    


