```mermaid
---
title: Reindeer Move
---
flowchart TD
    Start --> init[Initilize param and field]
    init --> checkDistanceReindeer[Check distance with other raindeer]
    checkDistanceReindeer --> avoidCollidsion[Check collision by protected_range^2 and add neighbor by visual_range]
    avoidCollidsion --> updateFore[Update forece by neighbors]
    updateFore --> hasPredator{Has Predator}
    hasPredator -- Yes --> findPredator[Calculate Predator Range]
    findPredator --> inRange{Predata in alert_range?}
    inRange -- Yes --> predatorUpdateForce[Update Predator Force]
    inRange -- No --> notFlee[Flee = No]
    notFlee --> findFoodLocation[Locate food location by visual range]
    predatorUpdateForce --> calculateTotalForce[Calculate Total Force, velocity, postion]
    hasPredator -- No --> findFoodLocation
    findFoodLocation --> calculateTotalForce
    calculateTotalForce --> updateVelocity [Update velocity]
    updateVelocity --> updatePosition[Update position]
    updatePosition --> updateBoundary[Update boundary reflection]
    updateBoundary --> decreaseEnergy[Decrease Energy]
    decreaseEnergy --> return[Return new value]