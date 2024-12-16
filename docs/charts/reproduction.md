```mermaid
---
title: Reindeer & Predator Reproduction
---
flowchart TD
    Start --> updateAge[Update Age]
    updateAge --> checkReproduction{Age > Reproduction age && energy > Reproduction energy}
    checkReproduction -- Yes --> findMate[Find near other parent]
    findMate --> setupOffspringPosition[Set offspring position]
    setupOffspringPosition --> createNewOffspring[Create New Offspring]
    createNewOffspring --> addToGroup[Add to reindeer or predator group]
    checkReproduction -- No --> return[Return]
    addToGroup --> return
