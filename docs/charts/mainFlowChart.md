```mermaid
---
title: Main
---
flowchart TD
    Start --> config[Read Config]
    config --> init[Initilize param and field]
    init --> create[Create Reindeer & Predator]
    create --> simulate[Simulate]
    simulate --> foodGrid[Update food]
    foodGrid --> reindeerAction[[Reindeer Action]]
    subgraph reindeerSubAction
        reindeerMove[Reindeer Move] --> reindeerGraze[Reindeer Graze]
        reindeerGraze --> reindeerDie[Reindeer die]
    end
    reindeerAction --> predatorAction[[Predator Action]]
    subgraph predatorSubAction
        predatorMove[Predator Move] --> predatorEat[Predator Eat]
        predatorEat --> predatorDie[Predator die]
    end
    predatorAction --> savePopulation(Save Population)
    savePopulation --> reproduction[[Reproduction]] 
    reproduction~~~|"Every reproduction_interval"|reproduction
    reproduction --> culling[Culling Reindeer]
    %% culling~~~|"Every reproduction_interval last 20 reindeer"|culling
    culling --> checkIntrusion[Check Intrusion]
    
    
    checkIntrusion --> visualization[visualization]
    visualization --> isStop{Reindeer == 0}
    isStop -- No --> simulate
    isStop -- Yes --> simulateStop[Stop simulate]
    simulateStop --> showResult[Show graph result]
    showResult --> endProcess[Finish Simulation]

    reindeerAction--oreindeerSubAction
    predatorAction --o predatorSubAction
    