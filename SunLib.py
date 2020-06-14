""" Sunlight library

Module to determine edges illuminated by a point light source on arbitrary
number of polygons and line segments in the 2D plane.

Dependancies: numpy, copy

Functions included:
    *TripletOrientation(point1, point2, point3)
    *IsRayIntersectLine(inputPoint, inputLine)
    *EdgeLen(edge)
    *EdgesLen(edge)
    *IsPointOnEdge(edge, point)
    *IsOnPolygon(inputPolygon, inputPoint)
    *IsInPolygon(inputPolygon, inputPoint)
    *CartPolar(x, y)
    *PolarCart(r, theta)
    *ShadowPolygon(edge, sun, max)
    *IsLinesIntersect(line1, line2)
    *IntersectionPoint(line1, line2)
    *FurthestFromSun(inputBuildings, sun)
    *IlluminatedLength(inputBuildings, sun)
    *GiveIlluminatedEdges(allEdges, sun)
    *GiveLitEdges(inputBuildings, sun)
    *BuildingsToEdges(inputBuildings)

Author : Bharath P. Kamath
contact: bharath.kamath705@gmail.com
"""

import numpy as np
import copy

def TripletOrientation(point1, point2, point3):
    """
    Tells if an ordered triplet of points are collinear, oriented clockwise or counter clockwise
    Based on cross product method for finding orientation of 3 points

    Parameters
    ----------
    point1: list of two floats
                x and y coordinates of a point

    point2: list of two floats
                x and y coordinates of a point

    point3: list of two floats
                x and y coordinates of a point

    Returns
    -------
    0 : the points are collinear
    1 : the points are oriented clockwise
    2 : the points are oriented counter clockwise
    """

    conditionVal = (point2[1] - point1[1] ) * (point3[0] - point2[0]) - (point3[1] - point2[1]) * (point2[0] - point1[0])

    # check if collinear
    if (conditionVal == 0):
        return 0

    #check if clockwise
    if (conditionVal > 0):
        return 1

    # check if counter clockwise
    if (conditionVal < 0 ):
        return 2

def IsRayIntersectLine(inputPoint, inputLine):
    """
    Return True if rightward ray from inputPoint intersects inputLine
    Assumes that the point does not lie on the line

    Parameters
    ----------
    inputPoint: array of floats (dim: 1 x 2)
                    co-ordinates of the point

    inputLine: 2D array of floats (dim: 2 x 2)
                    co-ordinates of line vertices

    Returns
    -------
    True : rightward ray from inputPoint intersects inputLine
    False: rightward ray from inputPoint does not intersect with inputlLine
    """
    # check if atleast one vertex of inputLine lies to the right of inputPoint
    if ((inputLine[0][0] > inputPoint[0]) or (inputLine[1][0] > inputPoint[0])):

        # check if one vertex lies above and one lies below input point
        # accunt for edge cases where one or both points lie on the ray
        if ((inputLine[0][1] >= inputPoint[1]) and inputLine[1][1] <= inputPoint[1]):

            # the inputPoint, higher point, and lower point will have clockwise orientation if the line is to the right
            if (TripletOrientation(inputPoint, inputLine[0], inputLine[1]) == 1):
                return True
            else:
                return False

        # check if one vertex lies above and one lies below input point
        elif ((inputLine[0][1] <= inputPoint[1] and inputLine[1][1] >= inputPoint[1])):

            # the inputPoint, higher point, and lower point will have clockwise orientation if the line is to the right
            if (TripletOrientation(inputPoint, inputLine[1], inputLine[0]) == 1):
                return True
            else:
                return False


        # both vertices lie above or below the inputPoint
        else:
            return False

    # both vertices lie to the left of inputPoint
    else:
        return False

def EdgeLen(edge):
    """
    Returns the length of an edge

    Parameters
    ----------
    edge: 2 x 2 list of floats
            each row represents x and y coordinates of a point

    Returns
    -------
    length of the edge (float)
    """
    p1 = edge[0]
    p2 = edge [1]

    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def EdgesLen(edges):
    """
    Returns the length of an edge

    Parameters
    ----------
    edges: N X 2 X 2 list of floats
            N edges each with 2 vertices having an x and y coordinate

    Returns
    -------
    length of the edges (float)
    """

    #calculate length of exposed edges
    Len = 0
    for edge in edges:
        Len += EdgeLen(edge)
    return Len



def IsPointOnEdge(edge, point):
    """
    Returns True if a point lies on an edge

    Parameters
    ----------
    edge: 2 X 2 list of floats
            each row represents x and y coordinates of a point

    Returns
    -------
    True:  point lies on the edge
    False: point does not lie on the edge
    """
    epsilon = 0.01 # tolerance

    # for point to lie on the line segment
    #  length of (vertex1 -> point -> vertex2) should be equal to length of (vertex1 -> vertex2)
    return abs(EdgeLen([edge[0], point]) + EdgeLen([point, edge[1]]) - EdgeLen(edge)) < epsilon

def IsOnPolygon(inputPolygon, inputPoint):
    """
    Returns True if a point lies ON THE BOUNDARY of a Polygon

    Parameters
    ----------
    inputPolygon:  N X 2 list of floats
                    represents the N vertices of the polygon in 2D space

    inputPoint:    list of 2 floats
                    point which is to be checked

    Returns
    -------
    True:  Point lies on the polygon
    False: Point does not lie on the polygon

    """
    edges = len(inputPolygon)

    for i in range(edges):
        edge = [inputPolygon[i], inputPolygon[(i+1)%edges]]
        if (IsPointOnEdge(edge, inputPoint)):
            return True

    return False


def IsInPolygon(inputPolygon, inputPoint):
    """
    Returns true if a point lies INSIDE a polygon

    Parameters
    ----------
    inputPolygon: 2D array of floats (Dim: N X 2)
                    coordinates of a polygon

    inputPoint: array of floats (Dim: 1 X 2)
                    coordinates of a point

    Returns
    -------
    True:  Point lies inside or on the boundary of the polygon
    False: Point lies outside the polygon
    """

    # point on polygon is counted as inside polygon
    if (IsOnPolygon(inputPolygon, inputPoint)):
        return True

    # Ray casting algorithm by counting intersections of rightward ray with edges
    edges = len(inputPolygon)
    intersections = 0

    # modulo operator so that last point pairs with first point in the array
    # count how many edges the input point intersects (ray casting algorithm)
    for i in range(edges):
        if (IsRayIntersectLine(inputPoint, [inputPolygon[i], inputPolygon[(i+1)%edges]])):

            # check edge cases: a vertex lies on the ray - skip count if second vertex lies on or above ray
            if ((inputPoint[1] == inputPolygon[i][1]) and (inputPoint[1] <= inputPolygon[(i+1)%edges][1])):
                continue

            # check edge cases: a vertex lies on the ray - skip count if second vertex lies on or above ray
            elif ((inputPoint[1] == inputPolygon[(i+1)%edges][1]) and (inputPoint[1] <= inputPolygon[i][1])):
                continue

            else:
                intersections += 1

    # point inside polygon if odd number of intersections
    if ((intersections % 2) == 0):
        return False
    else:
        return True

def CartPolar(x, y):
    """
    Returns equivalent polar coordinates for given cartesian coordinates

    Parameters
    ----------
    x: floats
        x coordinate of the point
    y: float
        y coordinate of the point

    Returns
    -------
    r: float
        radial distance of the point

    theta: float
            angle
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    return r, theta

def PolarCart(r, theta):
    """
    Returns equivalent cartesian coordinates for given polar coordinates

    Parameters
    ----------
    r: float
        radial distance of the point

    theta: float
             principal angle
    Returns
    -------
    x: floats
        x coordinate of the point
    y: float
        y coordinate of the point


    """
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return (x, y)

def ShadowPolygon(edge, sun, max):
    """
    Returns a polygon describing the shadow region of an unobstructed edge illuminated by the sun

    Parameters
    ----------
    edge: 2 X 2 list of floats
            each row represents a vertex of the edge in space

    sun:  list of 2 floats
            x and y coordinates of the sun

    max:  float
            extent to which shadow extends downstream of the edge

    Returns
    -------
    shadowPol: 4 X 2 list of floats
                shadow polygon is four sided, each row represents x and y coordinate of a vertex

    """
    xs, ys = [0, 0], [0, 0]

    for index, vertex in enumerate(edge):

        # convert edge coordinates to polar with sun at the origin
        r, theta = CartPolar((vertex[0] - sun[0]), (vertex[1] - sun[1]))
        r += max # move trailing edge of shadow sufficiently far

        # convert trailing edge of shadow back to cartesian w.r.t origin
        xs[index], ys[index] = PolarCart(r, theta)
        xs[index] += sun[0]
        ys[index] += sun[1]

    # create the shadow polygon
    shadowPol = [edge[0], [xs[0], ys[0]], [xs[1], ys[1]], edge[1]]

    return shadowPol


def IsLinesIntersect(line1, line2):
    """
    Returns true if the two line segments intersect each other

    Parameters
    ----------
    line1: 2 X 2 list of floats
            each row represents an endpoint of the line segment

    line2: 2 X 2 list of floats
            each row represents an endpoint of the line segment

    Returns
    -------
    True:  the two line segments intersect
    False: the two line segments do not intersect

    """

    # check based of the endpoints. behaviour when line segments are collinear undefined
    if (TripletOrientation(line1[0], line1[1], line2[0]) != TripletOrientation(line1[0], line1[1], line2[1])):
        if (TripletOrientation(line2[0], line2[1], line1[0]) != TripletOrientation(line2[0], line2[1], line1[1])):
            return True
        else:
            return False
    else:
        return False

def IntersectionPoint(line1, line2):
    """
    Returns the intersection point of two line segments
    Line segments are assumed to intersect
    Use in combination with "IsLinesIntersect()"

    Parameters
    ----------
    line1: 2 X 2 list of floats
            each row represents an endpoint of the line segment

    line2: 2 X 2 list of floats
            each row represents an endpoint of the line segment

    Returns
    -------
    P: list of 2 floats
            intersection point of line1 and line2
    """
    #  -mx + a*y = c
    # 'a' is artificially introduced to address infinite slope case
    # a is set to 1 for finite slope and to 0 for infinite slop

    # check if line1 is vertical
    if ((line1[1][0] - line1[0][0]) == 0):
        m1 = - 1
        a1 = 0
    else:
        m1 = (line1[1][1] - line1[0][1]) / (line1[1][0] - line1[0][0])
        a1 = 1

    # check if line2 is vertical
    if ((line2[1][0] - line2[0][0]) == 0):
        m2 = -1
        a2 = 0
    else:
        m2 = (line2[1][1] - line2[0][1]) / (line2[1][0] - line2[0][0])
        a2 =1

    c1 = a1 * line1[0][1] - m1 * line1[0][0]
    c2 = a2 * line2[0][1] - m2 * line2[0][0]

    A = np.array([[-m1, a1], [-m2, a2]])
    b = np.array([c1, c2])

    P = np.linalg.solve(A, b)

    return P.tolist()

def FurthestFromSun(inputBuildings, sun):
    """
    Returns the distance between the sun and the furthest building vertex
    also works witn edges

    Parameters
    ----------
    inputBuildings: N X V X 2 list of floats
                        N buildings with V number vertices having an x and y coordinate
                        works with list of edges without modification

    sun: list of two floats
            coordinate location of the sun

    Returns
    -------
    furthestDist: float
                    distance between the sun and the furthest building
    """

    furthestDist = 0

    for building in inputBuildings:
        for vertex in building:
            if (EdgeLen([sun, vertex]) > furthestDist):
                furthestDist = EdgeLen([sun, vertex])

    return furthestDist

def IlluminatedLength(inputBuildings, sun):
    """
    Returns the total length of sides of buildings illuminated by the sun

    Parameters
    ----------
    inputBuildings: N X V X 2 list of floats
                        N buildings with V number vertices having an x and y coordinate

    sun: list of two floats
            coordinate location of the sun

    Returns
    -------
    exposedLen: float
                    total length of building edges illuminated by the sun
    """
    # tolerance
    gamma = 0.001

    # length such that shadow polygon extends far enough
    buffer_factor = 1.2
    max = buffer_factor * FurthestFromSun(inputBuildings, sun)

    # number of edges that make up a building
    edges = len(inputBuildings[0])

    # reformat inputBuildings with with edges instead of vertices
    # allEdges[edge][point][x or y coordinate]
    allEdges = np.zeros((len(inputBuildings)*edges, 2, 2))
    allEdges = allEdges.tolist()

    for i, building in enumerate(inputBuildings):
        for j in range(edges):
            allEdges[(i*edges) + j] = [building[j], building[(j+1)%edges]]

    while(True):

        # for each edge, modify every other edge that falls in its shadow
        # This loop calculates the shadow for each edge
        initLength = len(allEdges)
        for j in range(initLength):
                shadow = ShadowPolygon(allEdges[j], sun, max)

                # this loop checks and modifies edges that fall in shadow
                for q in range(initLength):

                        #skip if edge is the one thats casting the shadow
                        if((q==j)):
                            continue

                        # check if entire edge is in the shadow
                        if (IsInPolygon(shadow, allEdges[q][0]) and IsInPolygon(shadow, allEdges[q][1])):
                            # make the vertecies co-incident so edge length is zero
                            allEdges[q][0] = allEdges[q][1]


                        # check if edge is partly in the shadow (edge intersects shaddow polygon)
                        else:
                            for index in range(len(shadow)):
                                if (IsLinesIntersect([shadow[index], shadow[(index+1)%len(shadow)]], allEdges[q])):
                                    newPoint = IntersectionPoint([shadow[index], shadow[(index+1)%len(shadow)]], allEdges[q])

                                    # skip if edge case: vertex and intersection point are co-incident
                                    # find the other intersection point
                                    if ( (EdgeLen([newPoint, allEdges[q][0]]) < gamma) or (EdgeLen([newPoint, allEdges[q][1]]) < gamma) ):
                                        continue

                                    # check if first vertex in shadow
                                    if (IsInPolygon(shadow, allEdges[q][0])):
                                        allEdges[q][0] = newPoint
                                        break

                                    # check if second vertex in shadow
                                    elif (IsInPolygon(shadow, allEdges[q][1])):
                                            allEdges[q][1] = newPoint
                                            break

                                    # both vertices are outside the shadow but edge intersects shadow:
                                    else:
                                        # break into two edges: vertex1 to intersection point
                                        #   and intersection point to vertex 2
                                        #   one of these new edges will have a vertex in the shadow
                                        #   and will be modified in the next iteration
                                        allEdges.append([newPoint, allEdges[q][0]])
                                        allEdges[q][0] = newPoint

        if (len(allEdges) == initLength):
            break

    #calculate length of exposed edges
    exposedLen = 0
    for edge in allEdges:
        exposedLen += EdgeLen(edge)
    return exposedLen


def GiveIlluminatedEdges(inputEdges, sun):
    """
    Returns list of illuminated edges

    Parameters
    ----------
    allEdges: N X 2 X 2 list of floats
                        N edges with 2 vertices each with an x and y coordinate

    sun: list of two floats
            coordinate location of the sun

    Returns
    -------
    litEdges: N X 2 X 2 list of floats
                        N edges with 2 vertices each with an x and y coordinate
    """

    # tolerance
    gamma = 0.001
    # so that input is not modified due to pass by reference
    allEdges = copy.deepcopy(inputEdges)

    # length such that shadow polygon extends far enough
    buffer_factor = 1.2
    max = buffer_factor * FurthestFromSun(allEdges, sun)

    while(True):
        # for each edge, modify every other edge that falls in its shadow
        # This loop calculates the shadow for each edge
        initLength = len(allEdges)
        for j in range(initLength):
                shadow = ShadowPolygon(allEdges[j], sun, max)

                # this loop checks and modifies edges that fall in shadow
                for q in range(initLength):

                        #skip if edge is the one thats casting the shadow
                        if((q==j)):
                            continue

                        # check if entire edge is in the shadow
                        if (IsInPolygon(shadow, allEdges[q][0]) and IsInPolygon(shadow, allEdges[q][1])):
                            # make the vertecies co-incident so edge length is zero
                            allEdges[q][0] = allEdges[q][1]


                        # check if edge is partly in the shadow (edge intersects shaddow polygon)
                        else:
                            for index in range(len(shadow)):
                                if (IsLinesIntersect([shadow[index], shadow[(index+1)%len(shadow)]], allEdges[q])):
                                    newPoint = IntersectionPoint([shadow[index], shadow[(index+1)%len(shadow)]], allEdges[q])

                                    # skip if edge case: vertex and intersection point are co-incident
                                    # find the other intersection point
                                    if ( (EdgeLen([newPoint, allEdges[q][0]]) < gamma) or (EdgeLen([newPoint, allEdges[q][1]]) < gamma) ):
                                        continue

                                    # check if first vertex in shadow
                                    if (IsInPolygon(shadow, allEdges[q][0])):
                                        allEdges[q][0] = newPoint
                                        break

                                    # check if second vertex in shadow
                                    elif (IsInPolygon(shadow, allEdges[q][1])):
                                            allEdges[q][1] = newPoint
                                            break

                                    # both vertices are outside the shadow but edge intersects shadow:
                                    else:
                                        # break into two edges: vertex1 to intersection point
                                        #   and intersection point to vertex 2
                                        #   one of these new edges will have a vertex in the shadow
                                        #   and will be modified in the next iteration
                                        allEdges.append([newPoint, allEdges[q][0]])
                                        allEdges[q][0] = newPoint

        if (len(allEdges) == initLength):
            break

    litEdges = []
    delta = 0.1

    for i, edge in enumerate(allEdges):
        if ((EdgeLen(edge) > delta)):
             litEdges.append(edge)

    return  litEdges

def GiveLitEdges(inputBuildings, sun):
    """
    Returns the edges illuminated by the sun

    Parameters
    ----------
    inputBuildings: N X V X 2 list of floats
                        N buildings with V number vertices having an x and y coordinate

    sun: list of two floats
            coordinate location of the sun

    Returns
    -------
    exposedLen: float
                    total length of building edges illuminated by the sun
    """
    # tolerance
    gamma = 0.001

    # length such that shadow polygon extends far enough
    buffer_factor = 1.2
    max = buffer_factor * FurthestFromSun(inputBuildings, sun)

    # number of edges that make up a building
    edges = len(inputBuildings[0])

    # reformat inputBuildings with with edges instead of vertices
    # allEdges[edge][point][x or y coordinate]
    allEdges = np.zeros((len(inputBuildings)*edges, 2, 2))
    allEdges = allEdges.tolist()

    for i, building in enumerate(inputBuildings):
        for j in range(edges):
            allEdges[(i*edges) + j] = [building[j], building[(j+1)%edges]]

    while(True):

        # for each edge, modify every other edge that falls in its shadow
        # This loop calculates the shadow for each edge
        initLength = len(allEdges)
        for j in range(initLength):
                shadow = ShadowPolygon(allEdges[j], sun, max)

                # this loop checks and modifies edges that fall in shadow
                for q in range(initLength):

                        #skip if edge is the one thats casting the shadow
                        if((q==j)):
                            continue

                        # check if entire edge is in the shadow
                        if (IsInPolygon(shadow, allEdges[q][0]) and IsInPolygon(shadow, allEdges[q][1])):
                            # make the vertecies co-incident so edge length is zero
                            allEdges[q][0] = allEdges[q][1]


                        # check if edge is partly in the shadow (edge intersects shaddow polygon)
                        else:
                            for index in range(len(shadow)):
                                if (IsLinesIntersect([shadow[index], shadow[(index+1)%len(shadow)]], allEdges[q])):
                                    newPoint = IntersectionPoint([shadow[index], shadow[(index+1)%len(shadow)]], allEdges[q])

                                    # skip if edge case: vertex and intersection point are co-incident
                                    # find the other intersection point
                                    if ( (EdgeLen([newPoint, allEdges[q][0]]) < gamma) or (EdgeLen([newPoint, allEdges[q][1]]) < gamma) ):
                                        continue

                                    # check if first vertex in shadow
                                    if (IsInPolygon(shadow, allEdges[q][0])):
                                        allEdges[q][0] = newPoint
                                        break

                                    # check if second vertex in shadow
                                    elif (IsInPolygon(shadow, allEdges[q][1])):
                                            allEdges[q][1] = newPoint
                                            break

                                    # both vertices are outside the shadow but edge intersects shadow:
                                    else:
                                        # break into two edges: vertex1 to intersection point
                                        #   and intersection point to vertex 2
                                        #   one of these new edges will have a vertex in the shadow
                                        #   and will be modified in the next iteration
                                        allEdges.append([newPoint, allEdges[q][0]])
                                        allEdges[q][0] = newPoint

        if (len(allEdges) == initLength):
            break

    litEdges = []
    delta = 0.1

    for i, edge in enumerate(allEdges):
        if ((EdgeLen(edge) > delta)):
             litEdges.append(edge)

    return  litEdges

def BuildingsToEdges(inputBuildings):

    # number of edges that make up a building
    edges = len(inputBuildings[0])

    # reformat inputBuildings with with edges instead of vertices
    # allEdges[edge][point][x or y coordinate]
    allEdges = np.zeros((len(inputBuildings)*edges, 2, 2))
    allEdges = allEdges.tolist()

    for i, building in enumerate(inputBuildings):
        for j in range(edges):
            allEdges[(i*edges) + j] = [building[j], building[(j+1)%edges]]

    return allEdges
