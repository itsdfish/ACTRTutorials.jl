;; (load "/home/dfish/.julia/dev/ACTRTutorial/actr6/load-act-r-6.lisp")
;; (load "/home/dfish/.julia/dev/ACTRTutorial/Tutorial_Models/Unit2/Addition/addition.lisp")
(clear-all)

(defvar *response*)
(defvar *response-time*)

;; num1 and num2 must be strings
(defun run-addition-model (num1 num2)
  (let ((window (open-exp-window "Count Model"
                                 :visible nil
                                 :width 600
                                 :height 300))
        (x 25))

    (reset)
    (install-device window)

    (dolist (text (list num1  num2))
      (add-text-to-exp-window :text text :x x :y 150 :width 75)
      (incf x 75))

    (setf *response* nil)
    (setf *response-time* nil)

    (proc-display)

    (run 30)
    (print *response*)
    (setf data (/ *response-time* 1000.0))))
   ;;  (if (string-equal *response* "j")
   ;;      (setf data (/ *response-time* 1000.0))
   ;;      (setf data nil))))

(defmethod rpm-window-key-event-handler ((win rpm-window) key)
  (setf *response-time* (get-time t))

(setf *response* (string key)))

(defun run-n-times (n num1 num2)
   (let ((rts (list)))
   (dotimes (i n)
      (push (list (run-count-model num1 num2)) rts))
      rts))

(define-model addition

(sgp :esc t :blc 1.5 :ans .3 :rt -10 :v t  :act nil :randomize-time t :VPFT t :VIDT t)

(chunk-type count-order current next)
(chunk-type add sum count)
(chunk-type number value name)
(chunk-type question num1 num2)

(add-dm
   (zero ISA number value "0" name "zero")
   (one ISA number value "1" name "one")
   (two ISA number value "2" name "two")
   (three ISA number value "3" name "three")
   (four ISA number value "4" name "four")
   (five ISA number value "5" name "five")
   (six ISA number value "6" name "six")
   (seven ISA number value "7" name "seven")
   (eight ISA number value "8" name "eight")
   (nine ISA number value "9" name "nine")
   (a ISA count-order current "0" next "1")
   (b ISA count-order current "1" next "2")
   (c ISA count-order current "2" next "3")
   (d ISA count-order current "3" next "4")
   (e ISA count-order current "4" next "5")
   (f ISA count-order current "5" next "6")
   (g ISA count-order current "6" next "7")
   (h ISA count-order current "7" next "8")
   (i ISA count-order current "8" next "9")
   (current-goal ISA add))

(P start
     =goal>
        ISA         add
    ?imaginal>
        state       free
        buffer      empty
    ?visual-location>
        buffer      unrequested
    ==>
    +imaginal>
        ISA         question
    +visual-location>
        ISA         visual-location
        screen-x    lowest
 )

 (P attend-visual-location
     =imaginal>
         ISA         question
    =visual-location>
        ISA         visual-location
    ?visual-location>
        buffer      requested
    ?visual>
        state       free
    ==>
    +visual>
        ISA         move-attention
        screen-pos  =visual-location
    =imaginal>
 )

 (P retrieve-meaning
     =visual>
        ISA         text
        value       =val
    ==>
     +retrieval>
         ISA        number
         value      =val
 )

 (P encode-start-number
    =retrieval>
        ISA        number
        value      =val
    =imaginal>
        ISA         question
        num1        nil
 ==>
    =imaginal>
        num1       =val
    +visual-location>
      ISA            visual-location
      :attended      nil
 )

 (P encode-end-number
     =goal>
        ISA         add
    =retrieval>
        ISA         number
        value       =val
    =imaginal>
        ISA         question
        num1       =num1
        num2         nil
 ==>
    =imaginal>
        num2        =val
    =goal>
       ; num1      =num1
       ; num2        =val
 )

(P initialize-addition
   =goal>
      ISA         add
      sum         nil
   =imaginal>
      ISA         question
      num1        =num1
      num2        =num2
==>
   =goal>
      sum         =num1
      count       "0"
   =imaginal>
   +retrieval>
      ISA        count-order
      current      =num1
)

(P terminate-addition
   =goal>
      ISA         add
      count       =num
      sum         =answer
   =imaginal>
      ISA        question
      num2        =num
   ?manual>
     state free
 ==>
   -goal>
  ; !output!       (=answer)
   +manual>
        ISA           press-key
        key           =answer
)

(P increment-count
   =goal>
      ISA         add
      sum         =sum ; 1
      count       =count ; 0
   =retrieval>
      ISA         count-order
      current       =count ;1 
      next          =newcount ;2
==>
   =goal>
      count       =newcount
   +retrieval>
      isa        count-order
      current    =sum
)

(P increment-sum
   =goal>
      ISA         add
      sum         =sum ; 1
      count       =count ; 0
   =imaginal>
      ISA         question
    - num2        =count ; 1 != 0
   =retrieval>
      ISA         count-order
      current     =sum ; 1 == 1
      next        =newsum ; 2
==>
   =goal>
      sum         =newsum
   =imaginal>
   +retrieval>
      ISA        count-order
      current    =count
)

(goal-focus current-goal)
)
