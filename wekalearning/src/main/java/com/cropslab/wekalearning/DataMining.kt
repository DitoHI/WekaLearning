package com.cropslab.wekalearning

import weka.classifiers.functions.LinearRegression
import weka.classifiers.functions.Logistic
import weka.core.Attribute
import weka.core.DenseInstance
import weka.core.Instances

abstract class DataMining(
    _labels: Array<String>
    , _trainingMapList: Map<String, DoubleArray>
    , _testList: DoubleArray
) {
    var labels: Array<String> = _labels
        private set
    var testList: DoubleArray = _testList
        private set
    var trainingMapList: Map<String, DoubleArray> = _trainingMapList
        private set
    var result = 0.0
        protected set
    protected var indexOfCondition = 0
    protected var rowSize = 0
    protected var colSize = 0
    protected lateinit var instance: Instances
    protected lateinit var predictionInstances: Instances
    protected lateinit var predictionDense: DenseInstance
    protected val attributes = ArrayList<Attribute>()

    open protected fun setAttributes() {
        if (this.labels.isNotEmpty()) {
            for (value in this.labels) {
                this.attributes.add(Attribute(value))
            }
            this.indexOfCondition = this.labels.size - 1
        }
    }

    open protected fun initializeSize() {
        if (this.trainingMapList.isNotEmpty()) {
            this.rowSize = 0
            this.trainingMapList.forEach {
                this.rowSize += it.value.size
            }
            this.colSize = this.labels.size
        }
    }

    protected fun initializeInstance() {
        this.instance = Instances("${labels.last()}Training", this.attributes, this.rowSize)
        this.instance.setClassIndex(this.indexOfCondition)

        this.predictionInstances = Instances("${labels.last()}Testing", this.attributes, 1)
        this.predictionDense = DenseInstance(this.colSize)

        for ((index, label) in labels.withIndex()) {
            if (index != labels.lastIndex) {
                this.predictionDense.setValue(this.predictionInstances.attribute(label), this.testList.get(index))
            }
        }
        this.predictionInstances.add(this.predictionDense)
        this.predictionInstances.setClassIndex(this.indexOfCondition)
    }

    open protected fun addInstance() {
        val rowPerColumnSize = this.rowSize / trainingMapList.keys.size
        for (i in 0 until rowPerColumnSize) {
            val dense = DenseInstance(this.colSize)
            for (label in labels) {
                dense.setValue(this.instance.attribute(label), trainingMapList[label]!!.get(i))
            }
            this.instance.add(dense)
        }
    }

    abstract fun evaluate()
}

class LRMining(
    _labels: Array<String>
    , _trainingMapList: Map<String, DoubleArray>
    , _testList: DoubleArray
): DataMining(_labels, _trainingMapList, _testList) {

    init {
        super.setAttributes()
        super.initializeSize()
        super.initializeInstance()
        super.addInstance()
        this.evaluate()
    }

    override fun evaluate() {
        val classifier = LinearRegression()
        classifier.buildClassifier(super.instance)
        super.result =  classifier.classifyInstance(super.predictionInstances.get(0))
    }
}

class LgRMining(
    _labels: Array<String>
    , _trainingMapList: Map<String, DoubleArray>
    , _testList: DoubleArray
    , _nominalTrainingList: Array<String>): DataMining(_labels, _trainingMapList, _testList) {
    var classPredicted: String = ""
        private set
    private var nominalTrainingList: Array<String>
    private var nominalLabel: Array<String>

    init {
        this.nominalTrainingList = _nominalTrainingList
        this.nominalLabel = _nominalTrainingList.distinct().toTypedArray()
        this.setAttributes()
        this.initializeSize()
        super.initializeInstance()
        this.addInstance()
        this.evaluate()
    }

    override fun setAttributes() {
        if (super.labels.isNotEmpty()) {
            for ((index, value) in super.labels.withIndex()) {
                if (index != super.labels.lastIndex) {
                    super.attributes.add(Attribute(value))
                } else {
                    super.attributes.add(Attribute(value, this.nominalLabel.toMutableList()))
                }
            }
            super.indexOfCondition = super.labels.size - 1
        }
    }

    override fun initializeSize() {
        super.initializeSize()
        super.rowSize += nominalTrainingList.size
    }

    override fun addInstance() {
        val rowPerColumnSize = this.rowSize / ( trainingMapList.keys.size + 1 )
        for (i in 0 until rowPerColumnSize) {
            val dense = DenseInstance(this.colSize)
            for ((index, label) in labels.withIndex()) {
                if (index == labels.lastIndex) {
                    dense.setValue(this.instance.attribute(label), nominalTrainingList.get(i))
                }
                else {
                    dense.setValue(this.instance.attribute(label), trainingMapList[label]!!.get(i))
                }
            }
            super.instance.add(dense)
        }
    }

    override fun evaluate() {
        val classifier = Logistic()
        classifier.buildClassifier(super.instance)
        super.result =  classifier.classifyInstance(super.predictionInstances.get(0))
        this.classPredicted = super.instance.classAttribute().value((super.result).toInt())
    }
}