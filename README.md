### Predicting and Classifying your Images with Weka

This library is trying to help you to do some basic machine learning to your data training and data testing with the help of Weka Library

### Inital Setup

* We use [WekaStripped](https://github.com/rjmarsan/Weka-for-Android) from another open source project to implement Linear and Logistic Regression to the data
* Install the dependency through gradle. Before that, add jitpack repository into the first of your `build.gradle`
```groovy
allprojects {
    repositories {
	    ...
	    maven { url 'https://jitpack.io' }
	}
}
```
Then add the dependency
```groovy
implementation 'com.github.DitoHI.weka-learning:app:0.1.0'
```

License
--------

    Copyright 2019 CropsLab, inc.
    
    Permission is hereby granted, free of charge, to any person obtaining a copy of 
    this software and associated documentation files (the "Software"), to deal in 
    the Software without restriction, including without limitation the rights to 
    use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
    of the Software, and to permit persons to whom the Software is furnished to do 
    so, subject to the following conditions:
    
    The above copyright notice and this permission notice shall be included in all 
    copies or substantial portions of the Software.
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS 
    FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
    COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER 
    IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION 
    WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
