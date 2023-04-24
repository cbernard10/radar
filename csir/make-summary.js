import { readdirSync, readFileSync, writeFile, writeFileSync } from 'fs'


;(() => {
    
    let new_summary = {}
    let files = readdirSync('./')
    let curr_summary

    if(files.includes('summary.json')){
        let data = readFileSync('./summary.json')
        curr_summary = JSON.parse(data)
    }

    files = files.filter(f => f.split('_').length === 3).filter(f => f.slice(f.length - 2) === 'FA')

    // make backup
    writeFile('./summary.json.old', JSON.stringify(curr_summary, null, 4),  () => {
        console.log('backup saved as summary.json.old')
    })

    // dataset indexing
    for(const dataset of files){
        let split = dataset.split('_')
        if (split.length !== 3){
            continue
        }
        new_summary[dataset] = {}
        let current_files = readdirSync(`./${dataset}`)
        new_summary[dataset]['files'] = current_files.filter(f => {
            let split = f.split('.')
            return ( split[split.length-1] === 'jpg' || split[split.length-1] === 'png')
        })

        if(current_files.includes(`${dataset}_summary.json`)){
            let raw_metadata = readFileSync(`./${dataset}/${dataset}_summary.json`)
            let data = JSON.parse(raw_metadata)
            let startDate = data.PCI.StartTime
            let stopDate = data.PCI.StopTime
            let comment = data.Info.Comment
            let windData = {names: data.EnvCon.Wind.Name, data: data.EnvCon.Wind.Spd, dir: data.EnvCon.Wind.Dir}
            let tempData = {names: data.EnvCon.Wind.Name, data: data.EnvCon.Temp.Data}
            let humidityData = {names: data.EnvCon.Humidity.Name, data: data.EnvCon.Humidity.Data}
            let rainData = {names: data.EnvCon.Rain.Name, data: data.EnvCon.Rain.Data}
            let waveData = {names: data.EnvCon.Wave.Name, data: data.EnvCon.Wave.HeightSignificant, dir: data.EnvCon.Wave.Dir}
            let pri = data.Waveform.PRI
            new_summary[dataset]['startTime'] = startDate
            new_summary[dataset]['stopTime'] = stopDate
            new_summary[dataset]['comment'] = comment
            new_summary[dataset]['windData'] = windData
            new_summary[dataset]['tempData'] = tempData
            new_summary[dataset]['humidityData'] = humidityData
            new_summary[dataset]['rainData'] = rainData
            new_summary[dataset]['waveData'] = waveData
            new_summary[dataset]['pri'] = pri
        }

        if(current_files.includes('burg')){
            const current_burg_files = readdirSync(`./${dataset}/burg`)
            new_summary[dataset]['burg'] = current_burg_files.filter(f => f !== 'umap' && !(/reg/.test(f)))
            new_summary[dataset]['burg_reg'] = current_burg_files.filter(f => f !== 'umap' && /reg/.test(f))
            // if(current_burg_files.includes('umap')){
            //     const current_umap_folders = readdirSync(`./${dataset}/burg/umap`).filter(f => f.split('.').length === 1)

            //     // new_summary[dataset]['umap'] = {}

            //     new_summary[dataset]['umap'] = readdirSync(`./${dataset}/burg/umap/barb/`)

            //     for(let folder of current_umap_folders){
            //         new_summary[dataset]['umap'][folder] = readdirSync(`./${dataset}/burg/umap/${folder}`)
            //         console.log(dataset+ ' ' + folder)
            //         console.log(new_summary[dataset]['umap'])
            //     }
            // }

            if(current_burg_files.includes('umap')){

                new_summary[dataset]['umap'] = readdirSync(`./${dataset}/burg/umap/`).filter(fname => fname != 'hdbscan.jpg')
                new_summary[dataset]['hdbscan'] = `./${dataset}/burg/umap/hdbscan.jpg`

            }
        }

        // if(curr_summary[dataset] && Object.keys(curr_summary[dataset]).includes('has_target')){
        //     new_summary[dataset]['has_target'] = curr_summary[dataset]['has_target']
        // }

        // else{
        //     const type = dataset[dataset.length-5]
        //     switch(type){
        //         case 'T':
        //             new_summary[dataset]['has_target'] = true
        //             break
        //         case 'N':
        //             new_summary[dataset]['has_target'] = true
        //             break
        //         case 'C':
        //             new_summary[dataset]['has_target'] = false
        //             break
        //     }
        // }

        // if(curr_summary[dataset] && Object.keys(curr_summary[dataset]).includes('ignore')){
        //     new_summary[dataset]['ignore'] = curr_summary[dataset]['ignore']
        // }
    }

    // writeFileSync('./summary.json', JSON.stringify(new_summary, null, 4),  () => {
    //     console.log('written')
    // })

    writeFileSync('./summary.json', JSON.stringify(new_summary, null, 4),  () => {
        console.log('written')
    })

    return new_summary
})()

// console.log(make_new_summary(summary)['00_005_TTrFA'])