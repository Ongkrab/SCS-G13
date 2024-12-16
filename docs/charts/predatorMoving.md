```mermaid
---
title: Reindeer Move
---
flowchart TD
    Start --> checkEnergy{Energy less than threshold?}
    checkEnergy -- Yes --> findPrey[Find prey in Visual Range]
    findPrey --> foundPrey{Found prey?}
    foundPrey -- Yes --> identfyTarget[Locate Prey direction]
    identfyTarget --> increaseSpeed[Increase velocity by hunt speed]
    foundPrey -- No --> increaseSpeed   
    checkEnergy -- No --> updateVelocity[Update velocity]
    updateVelocity --> updatePosition[Update position]
    increaseSpeed --> updatePosition
    updatePosition --> updateBoundary[Update boundary reflection]
    updateBoundary --> decreaseEnergy[Decrease Energy]
    decreaseEnergy --> return[Return new value]
